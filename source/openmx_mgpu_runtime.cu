#include "openmx_gpu_backend.h"
#include "openmx_env.h"
#include "openmx_gpu_architecture.h"
#define OPENMX_MGPU_RUNTIME_IMPLEMENTATION
/* Single-node multi-GPU admission and measured dispatch for cluster/Band.
   Worker backends contain no MPI. All cross-rank admission uses GPU UUID
   locks, avoiding MPI collectives inside dense-owner-only call sites. */
#include "openmx_mgpu_runtime.h"
#include "openmx_magma_mgpu.h"
#include "openmx_mgpu_gemm.h"
#include "openmx_dense_cpu.h"
#include "openmx_stdout_level.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <tuple>
#include <thread>
#include <unistd.h>
#include <vector>
#if OPENMX_GPU_HIP
#include <hsa/hsa_ext_amd.h>
#else
#if defined(__has_include)
#if __has_include(<nvml.h>)
#include <nvml.h>
#define OPENMX_MGPU_HAS_NVML_HEADER 1
#endif
#endif
#endif

extern "C" __attribute__((weak)) OpenMXGpuError OpenMX_GpuMemGetInfo(size_t *, size_t *);
#if OPENMX_GPU_HIP

extern "C" int openmx_gpu_physical_uuid(int device, unsigned char uuid[16])
{
    hipDeviceProp_t properties{};
    if (!uuid || OpenMXGpuGetDeviceProperties(&properties, device) != OPENMX_GPU_SUCCESS) {
        (void)OpenMXGpuGetLastError();
        return 0;
    }
    std::memcpy(uuid, properties.uuid.bytes, 16);
    return 1;
}
#else
#pragma weak openmx_mgpuGemmStagingBegin
#pragma weak openmx_mgpuGemmStagingEnd
#endif

namespace {
using Clock = std::chrono::steady_clock;
enum Backend { Native = 0, Magma = 1 };
int input_enabled = -1, input_aux_enabled = -1, input_gpu_limit = 0, topology_single_node = 0;
int input_force_magma = 0;
thread_local char host_error[512] = "";
thread_local int cluster_scope = 0;
thread_local int band_kpoints = 0, band_kpoint = -1;
std::timed_mutex operation_mutex;
using SolveKey = std::tuple<int,int,int,int,int,std::array<int,8>>;
std::map<SolveKey, Backend> solver_choices;
std::mutex topology_mutex;
std::map<std::vector<int>,bool> fast_link_choices;

int positive_env(int id, int fallback)
{
    const long x = openmx_env_long_at(id, fallback);
    return x >= 0 && x <= INT_MAX ? int(x) : fallback;
}

bool verbose()
{
    return positive_env(OPENMX_ENV_MGPU_VERBOSE, 0) != 0 || OPENMX_STDOUT_LEVEL(2);
}

const char *backend_name(Backend backend)
{
    return backend == Magma ? "magma" : "single";
}

/* No ordinal beyond CUDA_VISIBLE_DEVICES or scf.Gpu.Num is ever used.
   Automatic activation preserves the established consumer-GPU paths;
   explicit scf.mgpu.enable=on or OPENMX_MGPU=1 permits other devices too.
   A node-local pool (node_local=true) is the GPU set of the calling rank's
   own node also in a multi-node run: the host-backed capacity path (MAGMA
   solves and products as owner-only calls while the other ranks wait in
   MPI) and the work outside the eigensolver (the GPU workers of atom/grid
   local work, rank-owned services and the dense transforms of a cluster
   owner, and the cluster owner's cooperative matrix products), which never
   call MPI and only take node-local ordinals and locks.  The
   device-resident cooperative eigensolves and the Band k-point groups
   stay single-node. */
int pool_gpu_count(bool node_local, bool auxiliary = false)
{
    if ((!topology_single_node && !node_local) ||
        (auxiliary ? input_aux_enabled : input_enabled) == 0) return 0;
    if (!openmx_env_flag_at(OPENMX_ENV_GPU, 1)) return 0;
    if (openmx_env_flag_at(OPENMX_ENV_MGPU, -1) == 0) return 0;
    int count = 0;
    if (OpenMXGpuGetDeviceCount(&count) != OPENMX_GPU_SUCCESS) return 0;
    if (input_gpu_limit > 0) count = std::min(count, input_gpu_limit);
    count = std::min(count, positive_env(OPENMX_ENV_MGPU_NUM_GPUS, count));
    return count;
}

int configured_gpu_count() { return pool_gpu_count(false); }

#if !OPENMX_GPU_HIP
/* Optional time sharing retains the input k mesh on small GPU pools. Each
   operation holds the existing UUID locks, so overlapping owners serialize
   without changing rank-local CUDA/OpenACC device assignments. */
int band_groups(int total,int pool)
{
    if (total<1) return 0;
    return positive_env(OPENMX_ENV_MGPU_BAND_SHARED,0) ?
        std::min(total,std::max(1,pool/2)) : total;
}

#endif
bool allowed_gpu_range(int first,int count)
{
    const bool forced = input_enabled > 0 || openmx_env_flag_at(OPENMX_ENV_MGPU, -1) == 1;
    for (int i = first; i < first+count; ++i) {
#if OPENMX_GPU_HIP
        if (forced) continue;
        hipDeviceProp_t prop{};
        if (OpenMXGpuGetDeviceProperties(&prop, i) != OPENMX_GPU_SUCCESS) {
            (void)OpenMXGpuGetLastError();
            return false;
        }
        /* The CUDA throughput-ratio attribute has no equivalent on AMD.
           Match known Instinct architectures; consumer defaults stay local. */
        if (!openmx_amd_high_throughput_fp64(prop.gcnArchName)) return false;
#else
        int ratio = 0;
        if (!forced && (cudaDeviceGetAttribute(&ratio,
            cudaDevAttrSingleToDoublePrecisionPerfRatio,i) != OPENMX_GPU_SUCCESS || ratio <= 0 || ratio >= 8)) return false;
#endif
    }
    return true;
}

int band_group(int total,int index,int *devices,int capacity,int *total_gpus=nullptr)
{
    const int g=configured_gpu_count();
    if (total_gpus) *total_gpus=g;
    if (total<1 || index<0 || index>=total) return 0;
#if OPENMX_GPU_HIP
    const int q=g/total;
#else
    const int groups=band_groups(total,g), q=g/groups;
#endif
    if (q<2 || q>8 || (devices && capacity<q)) return 0;
#if OPENMX_GPU_HIP
    if (devices) for (int j=0;j<q;++j) devices[j]=index*q+j;
#else
    if (devices) for (int j=0;j<q;++j) devices[j]=(index%groups)*q+j;
#endif
    return q;
}

int devices_for_operation(int *devices)
{
    if (!cluster_scope) return 0;
    if (band_kpoints) {
        int count=band_group(band_kpoints,band_kpoint,devices,8);
        return count && allowed_gpu_range(devices[0],count) ? count : 0;
    }
    const int count=std::min(configured_gpu_count(),8);
    if (count<2 || !allowed_gpu_range(0,count)) return 0;
    for (int i=0;i<count;++i) devices[i]=i;
    return count;
}

/* Keep diagnostics local to the calling host thread. Device ordinals are
   relative to the CUDA-visible allocation, as are the Band worker lists. */
const char *operation_context(const int *devices,int count)
{
    static thread_local char text[384];
    char list[128] = "";
    size_t used=0;
    for (int i=0;i<count && used<sizeof(list);++i)
        used+=size_t(std::snprintf(list+used,sizeof(list)-used,"%s%d",i?",":"",devices[i]));
    int owner=-1;
    (void)OpenMXGpuGetDevice(&owner);
    std::snprintf(text,sizeof(text),
        "scope=%s kpoint=%d kpoints=%d group_gpus=%d used_gpus=%d devices=%s owner=%d",
        band_kpoints?"band":"cluster",band_kpoint,band_kpoints,count,
#if OPENMX_GPU_HIP
        band_kpoints?count*band_kpoints:count,count?list:"none",owner);
#else
        band_kpoints?count*band_groups(band_kpoints,configured_gpu_count()):count,count?list:"none",owner);
#endif
    return text;
}

/* The automatic GEMM crossover was measured with direct A100 NVLink paths.
   PCIe/host-staged transfers remain available through explicit on, but a
   size threshold alone cannot predict their benefit. Unknown topologies,
   including NVSwitch paths not positively identified here, decline auto. */
bool direct_link_query(const std::vector<int> &devices)
{
#if OPENMX_GPU_HIP
    for (size_t i=0;i<devices.size();++i) for (size_t j=0;j<devices.size();++j) {
        if (i==j) continue;
        int peer=0;
        uint32_t type=0,hops=0;
        if (OpenMXGpuDeviceCanAccessPeer(&peer,devices[i],devices[j])!=OPENMX_GPU_SUCCESS || !peer ||
            hipExtGetLinkTypeAndHopCount(devices[i],devices[j],&type,&hops)!=OPENMX_GPU_SUCCESS ||
            type!=HSA_AMD_LINK_INFO_TYPE_XGMI || hops!=1) {
            (void)OpenMXGpuGetLastError();
            return false;
        }
    }
    return true;
#else
#ifdef OPENMX_MGPU_HAS_NVML_HEADER
    struct NvmlSession {
        void *library=nullptr;
        decltype(&nvmlShutdown) shutdown=nullptr;
        bool initialized=false;
        ~NvmlSession() {
            if (initialized) (void)shutdown();
            if (library) (void)dlclose(library);
        }
    } session;
    session.library=dlopen("libnvidia-ml.so.1",RTLD_LAZY|RTLD_LOCAL);
    if (!session.library) return false;
    auto init=reinterpret_cast<decltype(&nvmlInit_v2)>(dlsym(session.library,"nvmlInit_v2"));
    session.shutdown=reinterpret_cast<decltype(&nvmlShutdown)>(dlsym(session.library,"nvmlShutdown"));
    auto handle=reinterpret_cast<decltype(&nvmlDeviceGetHandleByPciBusId_v2)>(dlsym(session.library,"nvmlDeviceGetHandleByPciBusId_v2"));
    auto state=reinterpret_cast<decltype(&nvmlDeviceGetNvLinkState)>(dlsym(session.library,"nvmlDeviceGetNvLinkState"));
    auto remote=reinterpret_cast<decltype(&nvmlDeviceGetNvLinkRemotePciInfo_v2)>(dlsym(session.library,"nvmlDeviceGetNvLinkRemotePciInfo_v2"));
    if (!init || !session.shutdown || !handle || !state || !remote || init()!=NVML_SUCCESS) return false;
    session.initialized=true;
    std::vector<cudaDeviceProp> prop(devices.size());
    std::vector<nvmlDevice_t> gpu(devices.size());
    for (size_t i=0;i<devices.size();++i) {
        if (OpenMXGpuGetDeviceProperties(&prop[i],devices[i])!=OPENMX_GPU_SUCCESS) return false;
        char pci[32];
        std::snprintf(pci,sizeof(pci),"%08x:%02x:%02x.0",prop[i].pciDomainID,prop[i].pciBusID,prop[i].pciDeviceID);
        if (handle(pci,&gpu[i])!=NVML_SUCCESS) return false;
    }
    for (size_t i=0;i<devices.size();++i) for (size_t j=0;j<devices.size();++j) {
        if (i==j) continue;
        int peer=0;
        if (OpenMXGpuDeviceCanAccessPeer(&peer,devices[i],devices[j])!=OPENMX_GPU_SUCCESS || !peer) return false;
        bool connected=false;
        for (unsigned link=0;link<NVML_NVLINK_MAX_LINKS;++link) {
            nvmlEnableState_t enabled=NVML_FEATURE_DISABLED;
            nvmlPciInfo_t pci{};
            if (state(gpu[i],link,&enabled)==NVML_SUCCESS && enabled==NVML_FEATURE_ENABLED &&
                remote(gpu[i],link,&pci)==NVML_SUCCESS &&
                pci.domain==unsigned(prop[j].pciDomainID) && pci.bus==unsigned(prop[j].pciBusID) &&
                pci.device==unsigned(prop[j].pciDeviceID)) { connected=true; break; }
        }
        if (!connected) return false;
    }
    return true;
#else
    (void)devices;
    return false;
#endif
#endif
}

bool fast_gemm_topology(const int *devices,int count)
{
    const char *mode=openmx_env_at(OPENMX_ENV_MGPU_GEMM);
    /* Only the explicitly requested legacy auto policy limits cooperation
       to NVLink. PCIe workers retain the bounded host staging path. */
    if (!mode || std::strcmp(mode,"auto")!=0) return true;
    int owner=-1;
    if (OpenMXGpuGetDevice(&owner)!=OPENMX_GPU_SUCCESS) return false;
    std::vector<int> key(devices,devices+count);
    key.push_back(owner);
    std::sort(key.begin(),key.end());
    key.erase(std::unique(key.begin(),key.end()),key.end());
    std::lock_guard<std::mutex> lock(topology_mutex);
    auto found=fast_link_choices.find(key);
    if (found!=fast_link_choices.end()) return found->second;
    const bool fast=direct_link_query(key);
    fast_link_choices[key]=fast;
    return fast;
}

/* Take locks in UUID order, not visibility order: two owners may use
   intersecting GPU subsets. Never unlink lock files while a waiter can
   hold the old inode. Process exit releases every OS lock automatically. */
class Admission {
    std::unique_lock<std::timed_mutex> guard;
    std::vector<int> fds;
public:
    bool ok = false;
    Admission(const int *devices, int count): guard(operation_mutex, std::defer_lock)
    {
        const auto deadline = Clock::now() + std::chrono::milliseconds(
            positive_env(OPENMX_ENV_GPU_LOCK_TIMEOUT_MS, 300000));
        if (!guard.try_lock_until(deadline)) {
            std::fprintf(stderr, "<OpenMX_MGPU> admission timed out waiting for the process lock.\n");
            return;
        }
        const char *directory = openmx_env_at(OPENMX_ENV_GPU_LOCK_DIR);
        if (!directory || !*directory) directory = "/tmp";
        if (directory[0] != '/') {
            std::fprintf(stderr, "<OpenMX_MGPU> OPENMX_GPU_LOCK_DIR must be an absolute shared node-local directory: %s\n", directory);
            return;
        }
        std::vector<std::string> names;
        int owner=-1;
        if (OpenMXGpuGetDevice(&owner)!=OPENMX_GPU_SUCCESS) return;
        /* Native calibration and output transfers also touch the caller's
           device, which may be outside the selected k-point worker group. */
        for (int i = 0; i <= count; ++i) {
#if OPENMX_GPU_HIP
            hipDeviceProp_t prop;
#else
            cudaDeviceProp prop;
#endif
            if (OpenMXGpuGetDeviceProperties(&prop,i==count?owner:devices[i]) != OPENMX_GPU_SUCCESS) return;
            char name[128];
            int used = std::snprintf(name,sizeof(name),".openmx-mgpu-%lu-",(unsigned long)getuid());
            for (unsigned char b : prop.uuid.bytes)
                used += std::snprintf(name+used,sizeof(name)-size_t(used),"%02x",unsigned(b));
            names.emplace_back(std::string(directory) + "/" + name);
        }
        std::sort(names.begin(),names.end());
        /* Opening the same inode twice and flocking both descriptors can
           self-deadlock, including when the owner is also a worker. */
        names.erase(std::unique(names.begin(),names.end()),names.end());
        /* A throwing constructor does not run this object's destructor.
           Finish every allocation before opening or locking a descriptor. */
        fds.reserve(names.size());
        for (const auto &name : names) {
            int fd = open(name.c_str(), O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW, 0600);
            if (fd < 0) {
                std::fprintf(stderr, "<OpenMX_MGPU> cannot open admission lock %s: %s\n",
                    name.c_str(), std::strerror(errno));
                return;
            }
            fds.push_back(fd);
            struct stat st;
            if (fstat(fd,&st) || st.st_uid != getuid() || !S_ISREG(st.st_mode)) {
                std::fprintf(stderr, "<OpenMX_MGPU> invalid admission lock ownership/type: %s\n", name.c_str());
                return;
            }
            while (flock(fd, LOCK_EX | LOCK_NB) != 0) {
                const int error = errno;
                if (error != EINTR && error != EWOULDBLOCK && error != EAGAIN) {
                    std::fprintf(stderr, "<OpenMX_MGPU> admission lock %s failed: %s\n",
                        name.c_str(), std::strerror(error));
                    return;
                }
                if (Clock::now() >= deadline) {
                    std::fprintf(stderr, "<OpenMX_MGPU> admission timed out for %s; "
                        "check OPENMX_GPU_LOCK_DIR and OPENMX_GPU_LOCK_TIMEOUT_MS.\n", name.c_str());
                    return;
                }
                std::this_thread::sleep_until(std::min(deadline,
                    Clock::now() + std::chrono::milliseconds(2)));
            }
        }
        ok = true;
    }
    ~Admission() { for (int fd : fds) { (void)flock(fd,LOCK_UN); (void)close(fd); } }
};

double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now()-start).count();
}

bool finite_eigenvalues(const std::vector<double> &w)
{
    for (size_t i=0; i<w.size(); ++i)
        if (!std::isfinite(w[i]) || (i && w[i]<w[i-1])) return false;
    return true;
}

/* Independent sampled FP64 residual and orthogonality checks protect the
   cached choice. Every eigenvector is checked for finite entries; samples
   cover both ends and interior of the requested spectrum. No eigenvector
   sign/phase comparison is used, so degenerate spectra remain valid. */
template<class T> bool validate_vectors(const std::vector<T>&,const std::vector<T>&,
                                      const std::vector<double>&,int,int,OpenMXBlasFillMode);

double realpart(double a) { return a; }
double realpart(OpenMXGpuComplex a) { return a.x; }
double imagpart(double) { return 0.0; }
double imagpart(OpenMXGpuComplex a) { return a.y; }
template<class T> T conjugate(T a) { return a; }
template<> OpenMXGpuComplex conjugate(OpenMXGpuComplex a) { return OpenMXGpuMakeComplex(a.x,-a.y); }
double diagonal_value(double a) { return a; }
OpenMXGpuComplex diagonal_value(OpenMXGpuComplex a) { return OpenMXGpuMakeComplex(a.x,0.0); }

template<class T>
T input_entry(const std::vector<T> &a,int n,int row,int col,OpenMXBlasFillMode uplo)
{
    if (row==col) return diagonal_value(a[size_t(col)*n+row]);
    return ((row>col)==(uplo==OPENMX_BLAS_FILL_LOWER)) ? a[size_t(col)*n+row]
                                                                  : conjugate(a[size_t(row)*n+col]);
}

template<class T>
bool finite_input(const std::vector<T> &a,int n,OpenMXBlasFillMode uplo)
{
    for (int col=0;col<n;++col) for (int row=col;row<n;++row) {
        const T value=input_entry(a,n,row,col,uplo);
        if (!std::isfinite(realpart(value)) || !std::isfinite(imagpart(value))) return false;
    }
    return true;
}

template<class T>
bool validate_vectors(const std::vector<T> &a, const std::vector<T> &q,
                      const std::vector<double> &w, int n, int nev,OpenMXBlasFillMode uplo)
{
    if (!finite_eigenvalues(w)) return false;
    for (size_t i=0;i<size_t(n)*nev;++i)
        if (!std::isfinite(realpart(q[i])) || !std::isfinite(imagpart(q[i]))) return false;
    const int samples=std::min(nev,5);
    std::array<int,5> cols{};
    for (int s=0;s<samples;++s) cols[s]=samples==1?0:int((long long)s*(nev-1)/(samples-1));
    const int rows=std::min(n,32);
    for (int sample=0;sample<samples;++sample) {
        const int j=cols[sample];
        long double err2=0, qnorm2=0, anorm2=0;
        for (int r=0;r<n;++r) {
            const T qv=q[size_t(j)*n+r];
            qnorm2+=(long double)realpart(qv)*realpart(qv)+(long double)imagpart(qv)*imagpart(qv);
        }
        for (int ri=0;ri<rows;++ri) {
            const int r=rows==1?0:int((long long)ri*(n-1)/(rows-1));
            long double re=0,im=0;
            for (int k=0;k<n;++k) {
                const T av=input_entry(a,n,r,k,uplo), qv=q[size_t(j)*n+k];
                if (!std::isfinite(realpart(av)) || !std::isfinite(imagpart(av))) return false;
                re+=(long double)realpart(av)*realpart(qv)-(long double)imagpart(av)*imagpart(qv);
                im+=(long double)realpart(av)*imagpart(qv)+(long double)imagpart(av)*realpart(qv);
                anorm2+=(long double)realpart(av)*realpart(av)+(long double)imagpart(av)*imagpart(av);
            }
            const T qv=q[size_t(j)*n+r];
            re-=(long double)w[j]*realpart(qv); im-=(long double)w[j]*imagpart(qv);
            err2+=re*re+im*im;
        }
        const long double scale=std::max(std::sqrt(anorm2)+(long double)std::fabs(w[j]),1e-300L)*std::sqrt(qnorm2);
        if (!(qnorm2>0) || std::sqrt(err2)>2e-10L*scale) return false;
        for (int other=0;other<samples;++other) {
            const int k=cols[other];
            long double re=0,im=0;
            for (int r=0;r<n;++r) {
                const T x=q[size_t(j)*n+r], y=q[size_t(k)*n+r];
                re+=(long double)realpart(x)*realpart(y)+(long double)imagpart(x)*imagpart(y);
                im+=(long double)realpart(x)*imagpart(y)-(long double)imagpart(x)*realpart(y);
            }
            re-=j==k?1.0L:0.0L;
            if (std::hypot(re,im)>2e-9L) return false;
        }
    }
    return true;
}

template<class T>
void canonicalize(std::vector<T> &a,int n,OpenMXBlasFillMode uplo)
{
    for (int j=0;j<n;++j)
        for (int i=j+1;i<n;++i)
            if (uplo==OPENMX_BLAS_FILL_LOWER) a[size_t(i)*n+j]=conjugate(a[size_t(j)*n+i]);
            else a[size_t(j)*n+i]=conjugate(a[size_t(i)*n+j]);
}

template<> void canonicalize(std::vector<OpenMXGpuComplex> &a,int n,OpenMXBlasFillMode uplo)
{
    for (int j=0;j<n;++j) {
        a[size_t(j)*n+j].y=0.0;
        for (int i=j+1;i<n;++i)
            if (uplo==OPENMX_BLAS_FILL_LOWER) a[size_t(i)*n+j]=conjugate(a[size_t(j)*n+i]);
            else a[size_t(j)*n+i]=conjugate(a[size_t(i)*n+j]);
    }
}

template<class T>
int solve_typed(OpenMXSolverHandle handle,OpenMXBlasFillMode uplo,int n,void *A,int lda,
                int nev,double *W,int64_t *meig,int *info,int ngpu,const int *devices,
                bool complex,openmx_mgpu_native_solve native_solve,void *native_context,
                OpenMXSolverStatus *status)
{
    const char *choice=openmx_env_at(OPENMX_ENV_MGPU_SOLVER);
    if (!choice || !*choice) choice="single";
    const bool automatic=std::strcmp(choice,"auto")==0;
    const bool force_magma=std::strcmp(choice,"magma")==0;
    if (!automatic && !force_magma) return 0;
    if (!openmx_magma_mgpu_available_devices(ngpu,devices)) return 0;
    int device=-1;
    if (OpenMXGpuGetDevice(&device)!=OPENMX_GPU_SUCCESS) return 0;
    std::array<int,8> group; group.fill(-1);
    std::copy_n(devices,ngpu,group.begin());
    const SolveKey key(n,nev,complex?1:0,ngpu,device,group);
    if (automatic) {
        /* A native winner needs no cooperative resources. Avoid taking
           every GPU's OS lock on later SCF steps, which would serialize
           otherwise independent single-GPU owners. A concurrent input
           reset can only make this a harmless native fallback. */
        std::lock_guard<std::timed_mutex> guard(operation_mutex);
        auto chosen=solver_choices.find(key);
        if (chosen!=solver_choices.end() && chosen->second==Native) return 0;
    }
    Admission admission(devices,ngpu);
    if (!admission.ok) return 0;
    OpenMXGpuStream stream=nullptr;
    if (OpenMXSolverGetStream(handle,&stream)!=OPENMX_SOLVER_SUCCESS) return 0;
    struct DeviceRestore { int dev; ~DeviceRestore() { (void)OpenMXGpuSetDevice(dev); } } restore{device};
    auto cached=solver_choices.find(key);
    const bool calibrate=automatic && cached==solver_choices.end();
    const int trials=calibrate ? std::max(1,std::min(3,positive_env(OPENMX_ENV_MGPU_TUNE_TRIALS,2))) : 1;
    Backend selected=Magma;
    if (automatic && !calibrate) selected=cached->second;
    if (automatic && !calibrate && selected==Native) return 0;

    /* Allocate everything that can throw before a native calibration
       modifies A. Trial buffers never alias the caller's device input. */
    const size_t count=size_t(n)*size_t(n), qcount=size_t(n)*size_t(nev);
    /* Best-result storage is calibration-only. Charge every allocation and
       copy required on an ordinary MGPU call to that candidate, while the
       native candidate measures its ordinary callback without MGPU costs. */
    std::vector<T> best(calibrate?qcount:0);
    std::vector<double> best_w(calibrate?size_t(nev):0,0.0);
    const auto preparation_start=Clock::now();
    std::vector<T> original(count), trial(count);
    std::vector<double> trial_w(size_t(nev),0.0);
    if (OpenMXGpuStreamSynchronize(stream)!=OPENMX_GPU_SUCCESS ||
        OpenMXGpuMemcpy2D(original.data(),size_t(n)*sizeof(T),A,size_t(lda)*sizeof(T),
                     size_t(n)*sizeof(T),n,OPENMX_GPU_COPY_DEVICE_TO_HOST)!=OPENMX_GPU_SUCCESS) return 0;
    /* A library can occasionally turn NaN input into finite eigenpairs of
       another matrix. Check the active triangle before calibration can
       accept even a nominally successful native result. */
    if (!finite_input(original,n,uplo)) {
        *status=OPENMX_SOLVER_INVALID_VALUE;
        return 1;
    }
    const double preparation_seconds=seconds_since(preparation_start);
    double best_seconds=std::numeric_limits<double>::infinity();
    Backend winner=Native;
    bool has_result=false, changed_device_input=false;

    auto publish=[&](const std::vector<T> &q,const std::vector<double> &w)->bool {
        const int zero=0;
        if (OpenMXGpuSetDevice(device)!=OPENMX_GPU_SUCCESS) return false;
        bool ok=OpenMXGpuMemcpy2DAsync(A,size_t(lda)*sizeof(T),q.data(),size_t(n)*sizeof(T),
                                  size_t(n)*sizeof(T),nev,OPENMX_GPU_COPY_HOST_TO_DEVICE,stream)==OPENMX_GPU_SUCCESS;
        if (ok) ok=OpenMXGpuMemcpyAsync(W,w.data(),size_t(nev)*sizeof(double),OPENMX_GPU_COPY_HOST_TO_DEVICE,stream)==OPENMX_GPU_SUCCESS;
        if (ok) ok=OpenMXGpuMemcpyAsync(info,&zero,sizeof(zero),OPENMX_GPU_COPY_HOST_TO_DEVICE,stream)==OPENMX_GPU_SUCCESS;
        /* Drain even after a later enqueue fails: host buffers stay alive
           until every successfully queued transfer has completed. */
        const bool drained=OpenMXGpuStreamSynchronize(stream)==OPENMX_GPU_SUCCESS;
        return ok && drained;
    };

    if (calibrate) {
        for (int repetition=0;repetition<trials;++repetition) {
            /* A second timing removes one-time library initialization from
               a repeated-SCF choice. Restoring the benchmark input is only
               calibration work, outside the native callback timing. */
            if (repetition && (OpenMXGpuSetDevice(device)!=OPENMX_GPU_SUCCESS ||
                OpenMXGpuMemcpy2DAsync(A,size_t(lda)*sizeof(T),original.data(),size_t(n)*sizeof(T),
                    size_t(n)*sizeof(T),n,OPENMX_GPU_COPY_HOST_TO_DEVICE,stream)!=OPENMX_GPU_SUCCESS ||
                OpenMXGpuStreamSynchronize(stream)!=OPENMX_GPU_SUCCESS)) {
                *status=OPENMX_SOLVER_EXECUTION_FAILED;
                return 1;
            }
            auto start=Clock::now();
            OpenMXSolverStatus native_status=OPENMX_SOLVER_EXECUTION_FAILED;
            try { native_status=native_solve(native_context); } catch (...) {}
            changed_device_input=true;
            int h_info=-1;
            const bool synchronized=OpenMXGpuStreamSynchronize(stream)==OPENMX_GPU_SUCCESS;
            const double elapsed=seconds_since(start);
            const bool valid=native_status==OPENMX_SOLVER_SUCCESS && synchronized &&
                OpenMXGpuMemcpy(&h_info,info,sizeof(h_info),OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS && h_info==0 &&
                (!meig || *meig==nev) &&
                OpenMXGpuMemcpy(trial_w.data(),W,size_t(nev)*sizeof(double),OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS &&
                finite_eigenvalues(trial_w) &&
                OpenMXGpuMemcpy2D(trial.data(),size_t(n)*sizeof(T),A,size_t(lda)*sizeof(T),
                             size_t(n)*sizeof(T),nev,OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS &&
                validate_vectors(original,trial,trial_w,n,nev,uplo);
            if (valid && elapsed<best_seconds) {
                std::copy_n(trial.begin(),qcount,best.begin());
                std::copy(trial_w.begin(),trial_w.end(),best_w.begin());
                best_seconds=elapsed; winner=Native; has_result=true;
            }
            if (verbose()) std::fprintf(stderr,"<OpenMX_MGPU> solver trial n=%d nev=%d type=%c gpus=%d backend=single seconds=%.6f valid=%d repetition=%d %s\n",
                n,nev,complex?'Z':'D',ngpu,elapsed,int(valid),repetition+1,operation_context(devices,ngpu));
            if (!valid) break;
        }
    }

    for (Backend candidate : {Magma}) {
        if (!calibrate && selected!=candidate) continue;
        for (int repetition=0;repetition<trials;++repetition) {
        auto start=Clock::now();
        std::copy(original.begin(),original.end(),trial.begin());
        canonicalize(trial,n,uplo);
        int rc=OPENMX_MAGMA_RESOURCES;
        try {
            rc=openmx_magma_mgpu_solve_host_devices(ngpu,devices,complex?1:0,1,1,n,1,nev,
                                                   trial.data(),n,trial_w.data());
        } catch (...) {
            /* A private host trial cannot damage the device input. */
        }
        bool valid=rc==0 && validate_vectors(original,trial,trial_w,n,nev,uplo);
        if (valid) {
            changed_device_input=true;
            if (!publish(trial,trial_w)) {
                *status=OPENMX_SOLVER_EXECUTION_FAILED;
                return 1; /* No retry after a partially committed output. */
            }
        }
        const double elapsed=preparation_seconds+seconds_since(start);
        if (verbose()) std::fprintf(stderr,"<OpenMX_MGPU> solver trial n=%d nev=%d type=%c gpus=%d backend=%s seconds=%.6f valid=%d code=%d repetition=%d %s\n",
            n,nev,complex?'Z':'D',ngpu,backend_name(candidate),elapsed,int(valid),rc,repetition+1,operation_context(devices,ngpu));
        if (valid && !calibrate) {
            /* The successful trial is already committed. Re-uploading it
               through a best-result buffer doubles the steady-state output
               traffic and would not be reflected in the calibration. */
            if (meig) *meig=nev;
            if (verbose()) std::fprintf(stderr,"<OpenMX_MGPU> solver selected n=%d nev=%d type=%c gpus=%d backend=%s seconds=%.6f calibration=0 %s\n",
                n,nev,complex?'Z':'D',ngpu,backend_name(candidate),elapsed,operation_context(devices,ngpu));
            *status=OPENMX_SOLVER_SUCCESS;
            return 1;
        }
        if (valid && elapsed<best_seconds) {
            std::copy_n(trial.begin(),qcount,best.begin());
            std::copy(trial_w.begin(),trial_w.end(),best_w.begin());
            best_seconds=elapsed; winner=candidate; has_result=true;
        }
        if (!valid) break;
        }
    }
    if (!has_result) {
        if (changed_device_input && (OpenMXGpuSetDevice(device)!=OPENMX_GPU_SUCCESS ||
            OpenMXGpuMemcpy2DAsync(A,size_t(lda)*sizeof(T),original.data(),size_t(n)*sizeof(T),
                size_t(n)*sizeof(T),n,OPENMX_GPU_COPY_HOST_TO_DEVICE,stream)!=OPENMX_GPU_SUCCESS ||
            OpenMXGpuStreamSynchronize(stream)!=OPENMX_GPU_SUCCESS)) {
            *status=OPENMX_SOLVER_EXECUTION_FAILED;
            return 1;
        }
        return 0;
    }
    if (!publish(best,best_w)) { *status=OPENMX_SOLVER_EXECUTION_FAILED; return 1; }
    if (meig) *meig=nev;
    if (automatic) {
        /* A failure to allocate a tiny tuning record cannot invalidate
           eigenvectors already computed and published successfully. */
        try { solver_choices[key]=winner; } catch (const std::bad_alloc &) {}
    }
    if (verbose()) std::fprintf(stderr,"<OpenMX_MGPU> solver selected n=%d nev=%d type=%c gpus=%d backend=%s seconds=%.6f calibration=%d %s\n",
        n,nev,complex?'Z':'D',ngpu,backend_name(winner),best_seconds,int(calibrate),operation_context(devices,ngpu));
    *status=OPENMX_SOLVER_SUCCESS;
    return 1;
}

/* A device-resident product always fits its owner's GPU (the operands are
   already there), so, like the eigensolver, it stays on that GPU.  Products
   use several GPUs only on the capacity rungs of the eigensolver: the
   host-backed MAGMA products on the GPUs of the owner's node, then the
   distributed gpusolver2 transforms over the GPUs of every node.
   Spreading a device-resident product over the permitted GPUs for speed is
   experimental and off by default: OPENMX_MGPU_GEMM=on (every size) or
   auto (the size/NVLink heuristic). */
bool gemm_eligible(int m,int n,int k)
{
    if (openmx_mgpuGemmInWorker() || m<=0 || n<=0 || k<=0) return false;
    const char *mode=openmx_env_at(OPENMX_ENV_MGPU_GEMM);
    if (!mode) return false;
    if (openmx_env_flag_at(OPENMX_ENV_MGPU_GEMM, -1)==1) return true;
    if (std::strcmp(mode,"auto")!=0) return false;
    const int min_n=positive_env(OPENMX_ENV_MGPU_GEMM_MIN_N,8192);
    const int min_output=min_n/2+min_n%2;
    return std::max(m,n)>=min_n && std::min(m,n)>=min_output && k>=min_n;
}

void report_gemm(int m,int n,int k,char type,const int *devices,int count)
{
    if (!verbose()) return;
    openmx_mgpu_gemm_report r{};
    openmx_mgpuGemmLastReport(&r);
    std::fprintf(stderr,"<OpenMX_MGPU> gemm type=%c m=%d n=%d k=%d gpus=%d tiles=%d seconds=%.6f reason=%s %s\n",
        type,m,n,k,r.active_gpus,r.tiles,r.seconds,r.reason?r.reason:"",operation_context(devices,count));
}

} // namespace

extern "C" void openmx_mgpu_configure(int enabled,int gpu_limit,int single_node)
{
    std::lock_guard<std::timed_mutex> guard(operation_mutex);
    input_enabled=enabled; input_aux_enabled=enabled; input_gpu_limit=gpu_limit; topology_single_node=single_node;
    solver_choices.clear();
    openmx_magma_mgpu_release_cache();
    std::lock_guard<std::mutex> topology_guard(topology_mutex);
    fast_link_choices.clear();
}
/* Auxiliary work has no k-point partition and remains available when the
   dense solver is ELPA/COSMA or the local k-point group has only one GPU. */
extern "C" void openmx_mgpu_configure_auxiliary(int enabled)
{
    input_aux_enabled=enabled;
}
extern "C" int openmx_mgpu_scope_enter(int allowed)
{
    const int previous=cluster_scope;
    cluster_scope=allowed?previous+1:0;
#if !OPENMX_GPU_HIP
    if(cluster_scope && openmx_mgpuGemmStagingBegin && openmx_mgpuGemmStagingEnd)
        openmx_mgpuGemmStagingBegin();
#endif
    return previous;
}
#if OPENMX_GPU_HIP
extern "C" void openmx_mgpu_scope_leave(int previous_scope) { cluster_scope=previous_scope; }
#else
extern "C" void openmx_mgpu_scope_leave(int previous_scope) {
    if(cluster_scope && openmx_mgpuGemmStagingBegin && openmx_mgpuGemmStagingEnd)
        openmx_mgpuGemmStagingEnd();
    cluster_scope=previous_scope;
}
#endif
extern "C" int openmx_mgpu_scope_active(void) { return cluster_scope!=0; }
/* The GPU workers' pool: every permitted GPU of this rank's node, also in
   a multi-node run (see pool_gpu_count). */
extern "C" int openmx_mgpu_pool_devices(int *devices,int capacity)
{
    const int count=pool_gpu_count(true,true);
    if (count<1 || !devices || capacity<count) return 0;
    for (int i=0;i<count;++i) devices[i]=i;
    return count;
}
extern "C" int openmx_mgpu_scope_devices(int *devices,int capacity)
{
    if (!cluster_scope) return 0;
    if (band_kpoints) return band_group(band_kpoints,band_kpoint,devices,capacity);
    const int count=pool_gpu_count(true);
    if (count<1 || !devices || capacity<count) return 0;
    for (int i=0;i<count;++i) devices[i]=i;
    return count;
}
extern "C" void *openmx_mgpu_admit_devices(const int *devices,int count)
{
    if (!devices || count<1 || count>64) return nullptr;
    int permitted=pool_gpu_count(true,true);
    if (count==1 && permitted==0) {
        /* Single-device callbacks also serve the original GPU paths when
           cooperation is disabled. They may only use the current device. */
        int owner=-1;
        if (OpenMXGpuGetDevice(&owner)!=OPENMX_GPU_SUCCESS || devices[0]!=owner ||
            OpenMXGpuGetDeviceCount(&permitted)!=OPENMX_GPU_SUCCESS) return nullptr;
        if (input_gpu_limit>0) permitted=std::min(permitted,input_gpu_limit);
    }
    for (int i=0;i<count;++i)
        if (devices[i]<0 || devices[i]>=permitted) return nullptr;
    try {
        Admission *admission=new Admission(devices,count);
        if (admission->ok) return admission;
        delete admission;
    } catch (const std::bad_alloc &) {}
    return nullptr;
}
extern "C" void openmx_mgpu_release_devices(void *admission)
{ delete static_cast<Admission*>(admission); }
extern "C" int openmx_mgpu_band_group(int total,int index,int *devices,int capacity)
{ return band_group(total,index,devices,capacity); }
extern "C" int openmx_mgpu_band_used_gpus(int total)
#if OPENMX_GPU_HIP
{ const int q=band_group(total,0,nullptr,0); return q?total*q:0; }
#else
{ const int q=band_group(total,0,nullptr,0); return q?band_groups(total,configured_gpu_count())*q:0; }
#endif
extern "C" openmx_mgpu_scope_state openmx_mgpu_band_scope_enter(int allowed,int total,int index)
{
    const openmx_mgpu_scope_state previous{cluster_scope,band_kpoints,band_kpoint};
    cluster_scope=allowed && total>0 && index>=0 && index<total ? 1 : 0;
#if !OPENMX_GPU_HIP
    if(cluster_scope && openmx_mgpuGemmStagingBegin && openmx_mgpuGemmStagingEnd)
        openmx_mgpuGemmStagingBegin();
#endif
    band_kpoints=total>0?total:-1; band_kpoint=index;
    if (verbose()) {
        int devices[8],g=0;
        const int q=band_group(total,index,devices,8,&g);
        const char *reason=!cluster_scope?"scope-disabled":q?"eligible":
            total>0 && g/total<2?"insufficient":total>0 && g/total>8?"backend-limit":"policy";
        std::fprintf(stderr,"<OpenMX_MGPU> band-group total_gpus=%d %s reason=%s\n",
            g,operation_context(devices,cluster_scope?q:0),reason);
    }
    return previous;
}
extern "C" void openmx_mgpu_band_scope_leave(openmx_mgpu_scope_state previous)
{
#if !OPENMX_GPU_HIP
    if(cluster_scope && openmx_mgpuGemmStagingBegin && openmx_mgpuGemmStagingEnd)
        openmx_mgpuGemmStagingEnd();
#endif
    cluster_scope=previous.active; band_kpoints=previous.band_kpoints; band_kpoint=previous.band_kpoint;
}

extern "C" int openmx_mgpu_solve_device_try(OpenMXSolverHandle handle,
    OpenMXSolverEigMode jobz,OpenMXBlasFillMode uplo,int64_t n,OpenMXGpuDataType type,
    void *A,int64_t lda,int64_t nev,double *W,int64_t *meig,int *info,
    openmx_mgpu_native_solve native_solve,void *native_context,OpenMXSolverStatus *status)
{
    if (!status || !handle || !A || !W || !info || !native_solve ||
        jobz!=OPENMX_SOLVER_EIG_VECTOR || n<=0 || n>INT_MAX || lda<n || lda>INT_MAX ||
        nev<1 || nev>n || (uplo!=OPENMX_BLAS_FILL_LOWER && uplo!=OPENMX_BLAS_FILL_UPPER) ||
        (type!=OPENMX_GPU_REAL64 && type!=OPENMX_GPU_COMPLEX64)) return 0;
    const char *mode=openmx_env_at(OPENMX_ENV_MGPU_SOLVER);
    if (mode && (std::strcmp(mode,"off")==0 || std::strcmp(mode,"single")==0)) return 0;
    const bool forced=mode && std::strcmp(mode,"magma")==0;
    if (!forced && n<positive_env(OPENMX_ENV_MGPU_SOLVER_MIN_N,4096)) return 0;
    int devices[8];
    const int count=devices_for_operation(devices);
    if (count<2) return 0;
    const size_t width=type==OPENMX_GPU_REAL64?sizeof(double):sizeof(OpenMXGpuComplex);
    if (uint64_t(n)>SIZE_MAX/width/uint64_t(n)) return 0;
    try {
        if (type==OPENMX_GPU_REAL64) return solve_typed<double>(handle,uplo,int(n),A,int(lda),int(nev),W,
            meig,info,count,devices,false,native_solve,native_context,status);
        return solve_typed<OpenMXGpuComplex>(handle,uplo,int(n),A,int(lda),int(nev),W,
            meig,info,count,devices,true,native_solve,native_context,status);
    } catch (const std::bad_alloc &) { return 0; }
#if OPENMX_GPU_HIP
}

extern "C" int openmx_mgpu_solve_host_result_try(int complex_matrix,int n,int nev,void *a,
    double *w,int *mout,int (*native_solve)(void *,double *),void *native_context,int *result)
{
    const char *mode=openmx_env_at(OPENMX_ENV_MGPU_SOLVER);
    if (!mode || (std::strcmp(mode,"auto") && std::strcmp(mode,"magma")) ||
        !a || !w || !native_solve || !result || n<=0 || nev<1 || nev>n) return 0;
    if (std::strcmp(mode,"magma") && n<positive_env(OPENMX_ENV_MGPU_SOLVER_MIN_N,4096)) return 0;
    int devices[8];
    if (devices_for_operation(devices)<2) return 0;
    std::vector<double> private_w;
    try {private_w.resize((size_t)n);} catch (const std::bad_alloc &) {return 0;}
    OpenMXSolverHandle handle=nullptr;
    double *dw=nullptr;
    int *di=nullptr;
    if (OpenMXSolverCreate(&handle)!=OPENMX_SOLVER_SUCCESS) return 0;
    if (OpenMXGpuMalloc((void**)&dw,(size_t)n*sizeof(double))!=OPENMX_GPU_SUCCESS ||
        OpenMXGpuMalloc((void**)&di,sizeof(int))!=OPENMX_GPU_SUCCESS) {
        (void)OpenMXGpuGetLastError();
        if (dw) OpenMXGpuFree(dw);
        OpenMXSolverDestroy(handle);
        return 0;
    }
    struct Native {
        int (*solve)(void *,double *); void *context; double *w; double *dw; int *di; int nev;
        static OpenMXSolverStatus run(void *opaque) {
            Native *c=static_cast<Native*>(opaque);
            int info=c->solve(c->context,c->w);
            if (OpenMXGpuMemcpy(c->di,&info,sizeof(info),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS)
                return HIPSOLVER_STATUS_INTERNAL_ERROR;
            if (info==0 && OpenMXGpuMemcpy(c->dw,c->w,(size_t)c->nev*sizeof(double),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS)
                return HIPSOLVER_STATUS_INTERNAL_ERROR;
            return info==0?OPENMX_SOLVER_SUCCESS:OPENMX_SOLVER_EXECUTION_FAILED;
        }
    } context{native_solve,native_context,private_w.data(),dw,di,nev};
    /* The callback has a private host eigenvalue destination and uploads it
       after the native solve, so the caller sees only the selected result. */
    OpenMXSolverStatus status=OPENMX_SOLVER_SUCCESS;
    int64_t found=nev;
    int handled=openmx_mgpu_solve_device_try(handle,OPENMX_SOLVER_EIG_VECTOR,OPENMX_BLAS_FILL_LOWER,
        n,complex_matrix?OPENMX_GPU_COMPLEX64:OPENMX_GPU_REAL64,a,n,nev,dw,&found,di,Native::run,&context,&status);
    if (handled) {
        int info=-1;
        if (status==OPENMX_SOLVER_SUCCESS && OpenMXGpuMemcpy(&info,di,sizeof(info),OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS &&
            info==0 && OpenMXGpuMemcpy(w,dw,(size_t)nev*sizeof(double),OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS) {
            *result=0; if (mout) *mout=(int)found;
        } else { *result=-1; if(mout)*mout=0; }
    }
    OpenMXGpuFree(di);OpenMXGpuFree(dw);OpenMXSolverDestroy(handle);
    return handled;
#endif
}

extern "C" int openmx_mgpu_dispatch_dgemm(OpenMXBlasHandle handle,
    OpenMXBlasOperation ta,OpenMXBlasOperation tb,int m,int n,int k,const double *alpha,
    const double *A,int lda,const double *B,int ldb,const double *beta,double *C,int ldc,
    OpenMXBlasStatus *status)
{
    if (!gemm_eligible(m,n,k)) return 0;
    int devices[64]; const int count=openmx_mgpu_scope_devices(devices,64);
    if (count<2) return 0;
    try {
        if (!fast_gemm_topology(devices,count)) return 0;
        Admission admission(devices,count);
        if (!admission.ok) return 0;
        const int handled=openmx_mgpuDgemmTry(handle,count,devices,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,status);
        report_gemm(m,n,k,'D',devices,count);
        return handled;
    } catch (const std::bad_alloc &) { return 0; }
}

extern "C" int openmx_mgpu_dispatch_zgemm(OpenMXBlasHandle handle,
    OpenMXBlasOperation ta,OpenMXBlasOperation tb,int m,int n,int k,const OpenMXGpuComplex *alpha,
    const OpenMXGpuComplex *A,int lda,const OpenMXGpuComplex *B,int ldb,
    const OpenMXGpuComplex *beta,OpenMXGpuComplex *C,int ldc,OpenMXBlasStatus *status)
{
    if (!gemm_eligible(m,n,k)) return 0;
    int devices[64]; const int count=openmx_mgpu_scope_devices(devices,64);
    if (count<2) return 0;
    try {
        if (!fast_gemm_topology(devices,count)) return 0;
        Admission admission(devices,count);
        if (!admission.ok) return 0;
        const int handled=openmx_mgpuZgemmTry(handle,count,devices,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,status);
        report_gemm(m,n,k,'Z',devices,count);
        return handled;
    } catch (const std::bad_alloc &) { return 0; }
}

extern "C" void openmx_mgpu_force_magma(int force) { input_force_magma=force!=0; }
extern "C" int openmx_mgpu_host_required(void)
{
    const char *mode=openmx_env_at(OPENMX_ENV_MGPU_MEMORY);
    return input_force_magma || (mode && std::strcmp(mode,"host")==0);
}
extern "C" const char *openmx_mgpu_host_error(void) { return host_error; }

extern "C" int openmx_mgpu_multi_node(void) { return !topology_single_node; }
extern "C" int openmx_mgpu_gpu_limit(void) { return input_gpu_limit; }

/* The limits of a host plan that need no GPU: the group size the pool
   allows and MAGMA's LP64 order.  Opens no CUDA context, so a caller can
   rule MAGMA out (with this reason) before any other test. */
extern "C" int openmx_mgpu_host_precheck(int total_kpoints,int n,int complex_matrix)
{
    host_error[0]='\0';
    const int pool=pool_gpu_count(total_kpoints==0);
#if OPENMX_GPU_HIP
    const int count=total_kpoints>0 ? pool/total_kpoints : std::min(pool,8);
#else
    const int count=total_kpoints>0 ? pool/band_groups(total_kpoints,pool) : std::min(pool,8);
#endif
    if (count<2 || count>8) {
        std::snprintf(host_error,sizeof(host_error),
            "MAGMA needs 2..8 GPUs per matrix: available pool=%d, scf.gpu.num limit=%d, kpoints=%d; GPU limit will not be exceeded",
            pool,input_gpu_limit,total_kpoints);
        return 0;
    }
    if (openmx_magma_mgpu_device_bytes(count,n,complex_matrix)==SIZE_MAX) {
        std::snprintf(host_error,sizeof(host_error),"unsupported MAGMA order/group: n=%d, gpus=%d (LP64 order limit 32766)",n,count);
        return 0;
    }
    return count;
}

extern "C" int openmx_mgpu_host_plan(int total_kpoints,int n,int complex_matrix)
{
    const int count=openmx_mgpu_host_precheck(total_kpoints,n,complex_matrix);
    if (count==0) return 0;
    /* Capacity recovery also applies to GeForce. The performance-oriented
       allowed_gpu_range filter must not forbid an otherwise feasible solve. */
    int owner=-1;
    if (OpenMXGpuGetDevice(&owner)!=OPENMX_GPU_SUCCESS) return 0;
    struct Restore { int dev; ~Restore() { (void)OpenMXGpuSetDevice(dev); } } restore{owner};
    const size_t required=openmx_magma_mgpu_device_bytes(count,n,complex_matrix);
#if OPENMX_GPU_HIP
    for (int group=0;group<std::max(1,total_kpoints);++group) {
#else
    const int groups=total_kpoints>0 ? band_groups(total_kpoints,pool_gpu_count(false)) : 1;
    for (int group=0;group<groups;++group) {
#endif
        int devices[8];
        for (int j=0;j<count;++j) devices[j]=group*count+j;
        if (!openmx_magma_mgpu_available_devices(count,devices)) {
            std::snprintf(host_error,sizeof(host_error),"MAGMA multi-GPU support is unavailable for the selected devices (build WITH_MAGMA=1)");
            return 0;
        }
        for (int j=0;j<count;++j) {
            size_t free_bytes=0,total=0;
            if (OpenMXGpuSetDevice(devices[j])!=OPENMX_GPU_SUCCESS || (OpenMX_GpuMemGetInfo ? OpenMX_GpuMemGetInfo(&free_bytes,&total) : OpenMXGpuMemGetInfo(&free_bytes,&total))!=OPENMX_GPU_SUCCESS) {
                std::snprintf(host_error,sizeof(host_error),"cannot query memory on GPU %d",devices[j]);
                return 0;
            }
            if (free_bytes<required) {
                std::snprintf(host_error,sizeof(host_error),
                    "MAGMA n=%d type=%c cannot fit: GPU %d free=%.3f GiB required=%.3f GiB, group=%d GPUs, scf.gpu.num limit=%d; GPU limit will not be exceeded",
                    n,complex_matrix?'Z':'D',devices[j],double(free_bytes)/(1ull<<30),double(required)/(1ull<<30),count,input_gpu_limit);
                return 0;
            }
        }
    }
    return count;
}

/* Host operations honor the same exact groups, but capacity recovery is
   independent of the automatic performance filter for consumer devices. */
static int host_devices(int *devices)
{
    if (!cluster_scope) return 0;
#if !OPENMX_GPU_HIP
    if (band_kpoints>0) return band_group(band_kpoints,band_kpoint,devices,8);
#endif
    const int pool=pool_gpu_count(band_kpoints==0);
    const int count=band_kpoints>0 ? pool/band_kpoints : std::min(pool,8);
    if (count<2 || count>8 || band_kpoints<0) return 0;
    for (int j=0;j<count;++j) devices[j]=(band_kpoints>0 ? band_kpoint*count:0)+j;
    return count;
}

extern "C" int openmx_mgpu_host_solve(int n,int nev,int complex_matrix,void *a,int lda,double *w)
{
    int devices[8], count=host_devices(devices);
    std::snprintf(host_error,sizeof(host_error),"no eligible MAGMA device group");
    int rc=OPENMX_MAGMA_RESOURCES;
    try {
        if (count>=2) {
            Admission admission(devices,count);
            if (admission.ok) {
                const auto start=Clock::now();
                rc=openmx_magma_mgpu_solve_host_devices(count,devices,complex_matrix,1,1,n,1,nev,a,lda,w);
                std::snprintf(host_error,sizeof(host_error),"MAGMA status=%d: %s",rc,openmx_magma_mgpu_last_error());
                if (verbose() || rc) std::fprintf(stderr,
                    "<OpenMX_MGPU> solver selected n=%d nev=%d type=%c gpus=%d backend=magma storage=host seconds=%.6f valid=%d %s%s%s\n",
                    n,nev,complex_matrix?'Z':'D',count,seconds_since(start),rc==0,operation_context(devices,count),rc?" error=":"",rc?host_error:"");
            }
        }
    } catch (const std::bad_alloc &) {
        std::snprintf(host_error,sizeof(host_error),"host memory allocation failed in MAGMA");
        rc=OPENMX_MAGMA_RESOURCES;
    }
    if (rc==OPENMX_MAGMA_SUCCESS) return 1;
    /* The adapter leaves A/W untouched on failure. Do not hide numerical
       validation failures or CUDA execution faults behind a CPU retry. */
    if (rc!=OPENMX_MAGMA_RESOURCES && rc!=OPENMX_MAGMA_UNAVAILABLE) return 0;
    std::fprintf(stderr,"<OpenMX_MGPU> n=%d: %s; using CPU LAPACK.\n",n,host_error);
    if (openmx_dense_cpu_eigen(n,nev,complex_matrix,a,lda,w)) return 1;
    std::snprintf(host_error,sizeof(host_error),"CPU eigensolver failed after GPU capacity fallback (n=%d, nev=%d)",n,nev);
    return 0;
}

static char cpu_transpose(OpenMXBlasOperation op)
{
    return op==OPENMX_BLAS_OP_N?'N':op==OPENMX_BLAS_OP_T?'T':op==OPENMX_BLAS_OP_C?'C':'?';
}

extern "C" int openmx_mgpu_host_dgemm(OpenMXBlasOperation ta,OpenMXBlasOperation tb,
    int m,int n,int k,const double *a,int lda,const double *b,int ldb,double *c,int ldc)
{
    int devices[8], count=host_devices(devices);
    const double alpha=1.0,beta=0.0;
    OpenMXBlasStatus status=OPENMX_BLAS_ALLOC_FAILED;
    int handled=0;
    try {
        if (count>=2) {
            Admission admission(devices,count);
            if (admission.ok) handled=openmx_mgpuDgemmHostTry(count,devices,ta,tb,m,n,k,&alpha,a,lda,b,ldb,&beta,c,ldc,&status);
            if (handled && status==OPENMX_BLAS_SUCCESS) {
                report_gemm(m,n,k,'D',devices,count); return 1;
            }
        }
    } catch (const std::bad_alloc &) {}
    openmx_mgpu_gemm_report report{}; openmx_mgpuGemmLastReport(&report);
    std::snprintf(host_error,sizeof(host_error),"host-backed multi-GPU DGEMM failed: %s (status=%d, group=%d, scf.gpu.num limit=%d)",
        count<2?"no eligible group":report.reason?report.reason:"resource or execution error",int(status),count,input_gpu_limit);
    if (handled) return 0; /* Failed publication is not a capacity decline. */
    std::fprintf(stderr,"<OpenMX_MGPU> %s; using CPU BLAS.\n",host_error);
    return openmx_dense_cpu_gemm(cpu_transpose(ta),cpu_transpose(tb),0,m,n,k,a,lda,b,ldb,c,ldc);
}
extern "C" int openmx_mgpu_host_zgemm(OpenMXBlasOperation ta,OpenMXBlasOperation tb,
    int m,int n,int k,const void *a,int lda,const void *b,int ldb,void *c,int ldc)
{
    int devices[8], count=host_devices(devices);
    const OpenMXGpuComplex alpha={1.0,0.0},beta={0.0,0.0};
    OpenMXBlasStatus status=OPENMX_BLAS_ALLOC_FAILED;
    int handled=0;
    try {
        if (count>=2) {
            Admission admission(devices,count);
            if (admission.ok) handled=openmx_mgpuZgemmHostTry(count,devices,ta,tb,m,n,k,&alpha,
                static_cast<const OpenMXGpuComplex*>(a),lda,static_cast<const OpenMXGpuComplex*>(b),ldb,&beta,
                static_cast<OpenMXGpuComplex*>(c),ldc,&status);
            if (handled && status==OPENMX_BLAS_SUCCESS) {
                report_gemm(m,n,k,'Z',devices,count); return 1;
            }
        }
    } catch (const std::bad_alloc &) {}
    openmx_mgpu_gemm_report report{}; openmx_mgpuGemmLastReport(&report);
    std::snprintf(host_error,sizeof(host_error),"host-backed multi-GPU ZGEMM failed: %s (status=%d, group=%d, scf.gpu.num limit=%d)",
        count<2?"no eligible group":report.reason?report.reason:"resource or execution error",int(status),count,input_gpu_limit);
    if (handled) return 0;
    std::fprintf(stderr,"<OpenMX_MGPU> %s; using CPU BLAS.\n",host_error);
    return openmx_dense_cpu_gemm(cpu_transpose(ta),cpu_transpose(tb),1,m,n,k,a,lda,b,ldb,c,ldc);
}
