/* Single-node multi-GPU admission and measured dispatch for cluster solves.
   Worker backends contain no MPI. All cross-rank admission uses GPU UUID
   locks, avoiding MPI collectives inside dense-owner-only call sites. */
#include "openmx_mgpu_runtime.h"
#include "openmx_magma_mgpu.h"
#include "openmx_mgpu_gemm.h"
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
#include <unistd.h>
#include <vector>
#if defined(__has_include)
#if __has_include(<nvml.h>)
#include <nvml.h>
#define OPENMX_MGPU_HAS_NVML_HEADER 1
#endif
#endif

extern "C" int openmx_paper_mgpu_available(void) __attribute__((weak));
extern "C" int openmx_paper_mgpu_dsyevdx_host(int,int,int,double*,int,double*) __attribute__((weak));
extern "C" int openmx_paper_mgpu_zheevdx_host(int,int,int,void*,int,double*) __attribute__((weak));
extern "C" const char *openmx_paper_mgpu_last_error(void) __attribute__((weak));

namespace {
using Clock = std::chrono::steady_clock;
enum Backend { Native = 0, Magma = 1, Paper = 2 };
int input_enabled = -1, input_gpu_limit = 0, topology_single_node = 0;
thread_local int cluster_scope = 0;
std::mutex operation_mutex;
using SolveKey = std::tuple<int,int,int,int,int>;
std::map<SolveKey, Backend> solver_choices;
std::mutex topology_mutex;
std::map<std::vector<int>,bool> nvlink_choices;

int positive_env(const char *name, int fallback)
{
    const char *s = std::getenv(name);
    if (!s || !*s) return fallback;
    char *end = nullptr;
    errno = 0;
    long x = std::strtol(s, &end, 10);
    return !errno && end && !*end && x >= 0 && x <= INT_MAX ? int(x) : fallback;
}

bool verbose()
{
    return positive_env("OPENMX_MGPU_VERBOSE", 0) != 0 || OPENMX_STDOUT_LEVEL(2);
}

const char *backend_name(Backend backend)
{
    return backend == Magma ? "magma" : backend == Paper ? "paper" : "single";
}

/* No ordinal beyond CUDA_VISIBLE_DEVICES or scf.Gpu.Num is ever used.
   Automatic activation preserves the established consumer-GPU paths;
   explicit scf.mgpu.enable=on or OPENMX_MGPU=1 permits other devices too. */
int devices_for_operation(int *devices)
{
    if (!cluster_scope || !topology_single_node || input_enabled == 0) return 0;
    const char *v = std::getenv("OPENMX_MGPU");
    if (v && *v && std::strcmp(v,"0") == 0) return 0;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    if (input_gpu_limit > 0) count = std::min(count, input_gpu_limit);
    count = std::min(count, positive_env("OPENMX_MGPU_NUM_GPUS", count));
    count = std::min(count, 8);
    if (count < 2) return 0;
    const bool forced = input_enabled > 0 || (v && std::strcmp(v,"1") == 0);
    for (int i = 0; i < count; ++i) {
        int ratio = 0;
        if (!forced && (cudaDeviceGetAttribute(&ratio,
            cudaDevAttrSingleToDoublePrecisionPerfRatio,i) != cudaSuccess || ratio <= 0 || ratio >= 8)) return 0;
        devices[i] = i;
    }
    return count;
}

/* The automatic GEMM crossover was measured with direct A100 NVLink paths.
   PCIe/host-staged transfers remain available through explicit on, but a
   size threshold alone cannot predict their benefit. Unknown topologies,
   including NVSwitch paths not positively identified here, decline auto. */
bool direct_nvlink_query(const std::vector<int> &devices)
{
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
        if (cudaGetDeviceProperties(&prop[i],devices[i])!=cudaSuccess) return false;
        char pci[32];
        std::snprintf(pci,sizeof(pci),"%08x:%02x:%02x.0",prop[i].pciDomainID,prop[i].pciBusID,prop[i].pciDeviceID);
        if (handle(pci,&gpu[i])!=NVML_SUCCESS) return false;
    }
    for (size_t i=0;i<devices.size();++i) for (size_t j=0;j<devices.size();++j) {
        if (i==j) continue;
        int peer=0;
        if (cudaDeviceCanAccessPeer(&peer,devices[i],devices[j])!=cudaSuccess || !peer) return false;
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
}

bool fast_gemm_topology(const int *devices,int count)
{
    const char *mode=std::getenv("OPENMX_MGPU_GEMM");
    if (mode && (std::strcmp(mode,"1")==0 || std::strcmp(mode,"on")==0)) return true;
    int owner=-1;
    if (cudaGetDevice(&owner)!=cudaSuccess) return false;
    std::vector<int> key(devices,devices+count);
    key.push_back(owner);
    std::sort(key.begin(),key.end());
    key.erase(std::unique(key.begin(),key.end()),key.end());
    std::lock_guard<std::mutex> lock(topology_mutex);
    auto found=nvlink_choices.find(key);
    if (found!=nvlink_choices.end()) return found->second;
    const bool fast=direct_nvlink_query(key);
    nvlink_choices[key]=fast;
    return fast;
}

/* Take locks in UUID order, not visibility order: two owners may use
   intersecting GPU subsets. Never unlink lock files while a waiter can
   hold the old inode. Process exit releases every OS lock automatically. */
class Admission {
    std::unique_lock<std::mutex> guard;
    std::vector<int> fds;
public:
    bool ok = false;
    Admission(const int *devices, int count): guard(operation_mutex)
    {
        std::vector<std::string> names;
        int owner=-1;
        if (cudaGetDevice(&owner)!=cudaSuccess) return;
        /* Native calibration and output transfers also touch the caller's
           device, which an explicit worker-count limit may have excluded. */
        for (int i = 0; i <= count; ++i) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop,i==count?owner:devices[i]) != cudaSuccess) return;
            char name[128];
            int used = std::snprintf(name,sizeof(name),"/tmp/.openmx-mgpu-%lu-",(unsigned long)getuid());
            for (unsigned char b : prop.uuid.bytes)
                used += std::snprintf(name+used,sizeof(name)-size_t(used),"%02x",unsigned(b));
            names.emplace_back(name);
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
            if (fd < 0) return;
            fds.push_back(fd);
            struct stat st;
            if (fstat(fd,&st) || st.st_uid != getuid() || !S_ISREG(st.st_mode)) return;
            int rc;
            do { rc = flock(fd,LOCK_EX); } while (rc && errno == EINTR);
            if (rc) return;
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
                                      const std::vector<double>&,int,int,cublasFillMode_t);

double realpart(double a) { return a; }
double realpart(cuDoubleComplex a) { return a.x; }
double imagpart(double) { return 0.0; }
double imagpart(cuDoubleComplex a) { return a.y; }
template<class T> T conjugate(T a) { return a; }
template<> cuDoubleComplex conjugate(cuDoubleComplex a) { return make_cuDoubleComplex(a.x,-a.y); }
double diagonal_value(double a) { return a; }
cuDoubleComplex diagonal_value(cuDoubleComplex a) { return make_cuDoubleComplex(a.x,0.0); }

template<class T>
T input_entry(const std::vector<T> &a,int n,int row,int col,cublasFillMode_t uplo)
{
    if (row==col) return diagonal_value(a[size_t(col)*n+row]);
    return ((row>col)==(uplo==CUBLAS_FILL_MODE_LOWER)) ? a[size_t(col)*n+row]
                                                                  : conjugate(a[size_t(row)*n+col]);
}

template<class T>
bool finite_input(const std::vector<T> &a,int n,cublasFillMode_t uplo)
{
    for (int col=0;col<n;++col) for (int row=col;row<n;++row) {
        const T value=input_entry(a,n,row,col,uplo);
        if (!std::isfinite(realpart(value)) || !std::isfinite(imagpart(value))) return false;
    }
    return true;
}

template<class T>
bool validate_vectors(const std::vector<T> &a, const std::vector<T> &q,
                      const std::vector<double> &w, int n, int nev,cublasFillMode_t uplo)
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
void canonicalize(std::vector<T> &a,int n,cublasFillMode_t uplo)
{
    for (int j=0;j<n;++j)
        for (int i=j+1;i<n;++i)
            if (uplo==CUBLAS_FILL_MODE_LOWER) a[size_t(i)*n+j]=conjugate(a[size_t(j)*n+i]);
            else a[size_t(j)*n+i]=conjugate(a[size_t(i)*n+j]);
}

template<> void canonicalize(std::vector<cuDoubleComplex> &a,int n,cublasFillMode_t uplo)
{
    for (int j=0;j<n;++j) {
        a[size_t(j)*n+j].y=0.0;
        for (int i=j+1;i<n;++i)
            if (uplo==CUBLAS_FILL_MODE_LOWER) a[size_t(i)*n+j]=conjugate(a[size_t(j)*n+i]);
            else a[size_t(j)*n+i]=conjugate(a[size_t(i)*n+j]);
    }
}

template<class T> int paper_solve(int,int,int,std::vector<T>&,std::vector<double>&) { return 1; }
template<> int paper_solve(int ngpu,int n,int nev,std::vector<double> &a,std::vector<double> &w)
{
    return openmx_paper_mgpu_dsyevdx_host ? openmx_paper_mgpu_dsyevdx_host(ngpu,n,nev,a.data(),n,w.data()) : 1;
}
template<> int paper_solve(int ngpu,int n,int nev,std::vector<cuDoubleComplex> &a,std::vector<double> &w)
{
    return openmx_paper_mgpu_zheevdx_host ? openmx_paper_mgpu_zheevdx_host(ngpu,n,nev,a.data(),n,w.data()) : 1;
}

template<class T>
int solve_typed(cusolverDnHandle_t handle,cublasFillMode_t uplo,int n,void *A,int lda,
                int nev,double *W,int64_t *meig,int *info,int ngpu,const int *devices,
                bool complex,openmx_mgpu_native_solve native_solve,void *native_context,
                cusolverStatus_t *status)
{
    const char *choice=std::getenv("OPENMX_MGPU_SOLVER");
    if (!choice || !*choice) choice="auto";
    const bool automatic=std::strcmp(choice,"auto")==0;
    const bool force_magma=std::strcmp(choice,"magma")==0;
    const bool force_paper=std::strcmp(choice,"paper")==0;
    if (!automatic && !force_magma && !force_paper) return 0;
    bool available_magma=openmx_magma_mgpu_available()!=0;
    bool available_paper=openmx_paper_mgpu_available && openmx_paper_mgpu_available()!=0 &&
        (complex ? openmx_paper_mgpu_zheevdx_host!=nullptr : openmx_paper_mgpu_dsyevdx_host!=nullptr);
    if ((!available_magma && !available_paper) || (force_magma && !available_magma) ||
        (force_paper && !available_paper)) return 0;
    int device=-1;
    if (cudaGetDevice(&device)!=cudaSuccess) return 0;
    const SolveKey key(n,nev,complex?1:0,ngpu,device);
    if (automatic) {
        /* A native winner needs no cooperative resources. Avoid taking
           every GPU's OS lock on later SCF steps, which would serialize
           otherwise independent single-GPU owners. A concurrent input
           reset can only make this a harmless native fallback. */
        std::lock_guard<std::mutex> guard(operation_mutex);
        auto chosen=solver_choices.find(key);
        if (chosen!=solver_choices.end() && chosen->second==Native) return 0;
    }
    Admission admission(devices,ngpu);
    if (!admission.ok) return 0;
    cudaStream_t stream=nullptr;
    if (cusolverDnGetStream(handle,&stream)!=CUSOLVER_STATUS_SUCCESS) return 0;
    struct DeviceRestore { int dev; ~DeviceRestore() { (void)cudaSetDevice(dev); } } restore{device};
    auto cached=solver_choices.find(key);
    const bool calibrate=automatic && cached==solver_choices.end();
    const int trials=calibrate ? std::max(1,std::min(3,positive_env("OPENMX_MGPU_TUNE_TRIALS",2))) : 1;
    Backend selected=force_paper?Paper:Magma;
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
    if (cudaStreamSynchronize(stream)!=cudaSuccess ||
        cudaMemcpy2D(original.data(),size_t(n)*sizeof(T),A,size_t(lda)*sizeof(T),
                     size_t(n)*sizeof(T),n,cudaMemcpyDeviceToHost)!=cudaSuccess) return 0;
    /* A library can occasionally turn NaN input into finite eigenpairs of
       another matrix. Check the active triangle before calibration can
       accept even a nominally successful native result. */
    if (!finite_input(original,n,uplo)) {
        *status=CUSOLVER_STATUS_INVALID_VALUE;
        return 1;
    }
    const double preparation_seconds=seconds_since(preparation_start);
    double best_seconds=std::numeric_limits<double>::infinity();
    Backend winner=Native;
    bool has_result=false, changed_device_input=false;

    auto publish=[&](const std::vector<T> &q,const std::vector<double> &w)->bool {
        const int zero=0;
        if (cudaSetDevice(device)!=cudaSuccess) return false;
        bool ok=cudaMemcpy2DAsync(A,size_t(lda)*sizeof(T),q.data(),size_t(n)*sizeof(T),
                                  size_t(n)*sizeof(T),nev,cudaMemcpyHostToDevice,stream)==cudaSuccess;
        if (ok) ok=cudaMemcpyAsync(W,w.data(),size_t(nev)*sizeof(double),cudaMemcpyHostToDevice,stream)==cudaSuccess;
        if (ok) ok=cudaMemcpyAsync(info,&zero,sizeof(zero),cudaMemcpyHostToDevice,stream)==cudaSuccess;
        /* Drain even after a later enqueue fails: host buffers stay alive
           until every successfully queued transfer has completed. */
        const bool drained=cudaStreamSynchronize(stream)==cudaSuccess;
        return ok && drained;
    };

    if (calibrate) {
        for (int repetition=0;repetition<trials;++repetition) {
            /* A second timing removes one-time library initialization from
               a repeated-SCF choice. Restoring the benchmark input is only
               calibration work, outside the native callback timing. */
            if (repetition && (cudaSetDevice(device)!=cudaSuccess ||
                cudaMemcpy2DAsync(A,size_t(lda)*sizeof(T),original.data(),size_t(n)*sizeof(T),
                    size_t(n)*sizeof(T),n,cudaMemcpyHostToDevice,stream)!=cudaSuccess ||
                cudaStreamSynchronize(stream)!=cudaSuccess)) {
                *status=CUSOLVER_STATUS_EXECUTION_FAILED;
                return 1;
            }
            auto start=Clock::now();
            cusolverStatus_t native_status=CUSOLVER_STATUS_EXECUTION_FAILED;
            try { native_status=native_solve(native_context); } catch (...) {}
            changed_device_input=true;
            int h_info=-1;
            const bool synchronized=cudaStreamSynchronize(stream)==cudaSuccess;
            const double elapsed=seconds_since(start);
            const bool valid=native_status==CUSOLVER_STATUS_SUCCESS && synchronized &&
                cudaMemcpy(&h_info,info,sizeof(h_info),cudaMemcpyDeviceToHost)==cudaSuccess && h_info==0 &&
                (!meig || *meig==nev) &&
                cudaMemcpy(trial_w.data(),W,size_t(nev)*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess &&
                finite_eigenvalues(trial_w) &&
                cudaMemcpy2D(trial.data(),size_t(n)*sizeof(T),A,size_t(lda)*sizeof(T),
                             size_t(n)*sizeof(T),nev,cudaMemcpyDeviceToHost)==cudaSuccess &&
                validate_vectors(original,trial,trial_w,n,nev,uplo);
            if (valid && elapsed<best_seconds) {
                std::copy_n(trial.begin(),qcount,best.begin());
                std::copy(trial_w.begin(),trial_w.end(),best_w.begin());
                best_seconds=elapsed; winner=Native; has_result=true;
            }
            if (verbose()) std::fprintf(stderr,"<OpenMX_MGPU> solver trial n=%d nev=%d type=%c gpus=%d backend=single seconds=%.6f valid=%d repetition=%d\n",
                n,nev,complex?'Z':'D',ngpu,elapsed,int(valid),repetition+1);
            if (!valid) break;
        }
    }

    for (Backend candidate : {Magma,Paper}) {
        if ((!calibrate && selected!=candidate) || (candidate==Magma && !available_magma) ||
            (candidate==Paper && !available_paper)) continue;
        for (int repetition=0;repetition<trials;++repetition) {
        auto start=Clock::now();
        std::copy(original.begin(),original.end(),trial.begin());
        canonicalize(trial,n,uplo);
        int rc=OPENMX_MAGMA_RESOURCES;
        try {
            rc=candidate==Magma ? openmx_magma_mgpu_solve_host(ngpu,complex?1:0,1,1,n,1,nev,
                                                              trial.data(),n,trial_w.data())
                                : paper_solve(ngpu,n,nev,trial,trial_w);
        } catch (...) {
            /* A private host trial cannot damage the device input. */
        }
        bool valid=rc==0 && validate_vectors(original,trial,trial_w,n,nev,uplo);
        if (valid) {
            changed_device_input=true;
            if (!publish(trial,trial_w)) {
                *status=CUSOLVER_STATUS_EXECUTION_FAILED;
                return 1; /* No retry after a partially committed output. */
            }
        }
        const double elapsed=preparation_seconds+seconds_since(start);
        if (verbose()) std::fprintf(stderr,"<OpenMX_MGPU> solver trial n=%d nev=%d type=%c gpus=%d backend=%s seconds=%.6f valid=%d code=%d repetition=%d\n",
            n,nev,complex?'Z':'D',ngpu,backend_name(candidate),elapsed,int(valid),rc,repetition+1);
        if (valid && !calibrate) {
            /* The successful trial is already committed. Re-uploading it
               through a best-result buffer doubles the steady-state output
               traffic and would not be reflected in the calibration. */
            if (meig) *meig=nev;
            if (verbose()) std::fprintf(stderr,"<OpenMX_MGPU> solver selected n=%d nev=%d type=%c gpus=%d backend=%s seconds=%.6f calibration=0\n",
                n,nev,complex?'Z':'D',ngpu,backend_name(candidate),elapsed);
            *status=CUSOLVER_STATUS_SUCCESS;
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
        if (changed_device_input && (cudaSetDevice(device)!=cudaSuccess ||
            cudaMemcpy2DAsync(A,size_t(lda)*sizeof(T),original.data(),size_t(n)*sizeof(T),
                size_t(n)*sizeof(T),n,cudaMemcpyHostToDevice,stream)!=cudaSuccess ||
            cudaStreamSynchronize(stream)!=cudaSuccess)) {
            *status=CUSOLVER_STATUS_EXECUTION_FAILED;
            return 1;
        }
        return 0;
    }
    if (!publish(best,best_w)) { *status=CUSOLVER_STATUS_EXECUTION_FAILED; return 1; }
    if (meig) *meig=nev;
    if (automatic) {
        /* A failure to allocate a tiny tuning record cannot invalidate
           eigenvectors already computed and published successfully. */
        try { solver_choices[key]=winner; } catch (const std::bad_alloc &) {}
    }
    if (verbose()) std::fprintf(stderr,"<OpenMX_MGPU> solver selected n=%d nev=%d type=%c gpus=%d backend=%s seconds=%.6f calibration=%d\n",
        n,nev,complex?'Z':'D',ngpu,backend_name(winner),best_seconds,int(calibrate));
    *status=CUSOLVER_STATUS_SUCCESS;
    return 1;
}

bool gemm_eligible(int m,int n,int k)
{
    if (openmx_mgpuGemmInWorker() || m<=0 || n<=0 || k<=0) return false;
    const char *mode=std::getenv("OPENMX_MGPU_GEMM");
    if (mode && (std::strcmp(mode,"0")==0 || std::strcmp(mode,"off")==0)) return false;
    if (mode && (std::strcmp(mode,"1")==0 || std::strcmp(mode,"on")==0)) return true;
    const int min_n=positive_env("OPENMX_MGPU_GEMM_MIN_N",8192);
    const int min_output=min_n/2+min_n%2;
    return std::max(m,n)>=min_n && std::min(m,n)>=min_output && k>=min_n;
}

void report_gemm(int m,int n,int k,char type)
{
    if (!verbose()) return;
    openmx_mgpu_gemm_report r{};
    openmx_mgpuGemmLastReport(&r);
    std::fprintf(stderr,"<OpenMX_MGPU> gemm type=%c m=%d n=%d k=%d gpus=%d tiles=%d seconds=%.6f reason=%s\n",
        type,m,n,k,r.active_gpus,r.tiles,r.seconds,r.reason?r.reason:"");
}

} // namespace

extern "C" void openmx_mgpu_configure(int enabled,int gpu_limit,int single_node)
{
    std::lock_guard<std::mutex> guard(operation_mutex);
    input_enabled=enabled; input_gpu_limit=gpu_limit; topology_single_node=single_node;
    solver_choices.clear();
    openmx_magma_mgpu_release_cache();
    std::lock_guard<std::mutex> topology_guard(topology_mutex);
    nvlink_choices.clear();
}
extern "C" int openmx_mgpu_scope_enter(int allowed)
{
    const int previous=cluster_scope;
    cluster_scope=allowed?previous+1:0;
    return previous;
}
extern "C" void openmx_mgpu_scope_leave(int previous_scope) { cluster_scope=previous_scope; }
extern "C" int openmx_mgpu_scope_active(void) { return cluster_scope!=0; }

extern "C" int openmx_mgpu_solve_device_try(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,cublasFillMode_t uplo,int64_t n,cudaDataType type,
    void *A,int64_t lda,int64_t nev,double *W,int64_t *meig,int *info,
    openmx_mgpu_native_solve native_solve,void *native_context,cusolverStatus_t *status)
{
    if (!status || !handle || !A || !W || !info || !native_solve ||
        jobz!=CUSOLVER_EIG_MODE_VECTOR || n<=0 || n>INT_MAX || lda<n || lda>INT_MAX ||
        nev<1 || nev>n || (uplo!=CUBLAS_FILL_MODE_LOWER && uplo!=CUBLAS_FILL_MODE_UPPER) ||
        (type!=CUDA_R_64F && type!=CUDA_C_64F)) return 0;
    const char *mode=std::getenv("OPENMX_MGPU_SOLVER");
    if (mode && (std::strcmp(mode,"off")==0 || std::strcmp(mode,"single")==0)) return 0;
    const bool forced=mode && (std::strcmp(mode,"magma")==0 || std::strcmp(mode,"paper")==0);
    if (!forced && n<positive_env("OPENMX_MGPU_SOLVER_MIN_N",4096)) return 0;
    int devices[8];
    const int count=devices_for_operation(devices);
    if (count<2) return 0;
    const size_t width=type==CUDA_R_64F?sizeof(double):sizeof(cuDoubleComplex);
    if (uint64_t(n)>SIZE_MAX/width/uint64_t(n)) return 0;
    try {
        if (type==CUDA_R_64F) return solve_typed<double>(handle,uplo,int(n),A,int(lda),int(nev),W,
            meig,info,count,devices,false,native_solve,native_context,status);
        return solve_typed<cuDoubleComplex>(handle,uplo,int(n),A,int(lda),int(nev),W,
            meig,info,count,devices,true,native_solve,native_context,status);
    } catch (const std::bad_alloc &) { return 0; }
}

extern "C" int openmx_mgpu_dispatch_dgemm(cublasHandle_t handle,
    cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const double *alpha,
    const double *A,int lda,const double *B,int ldb,const double *beta,double *C,int ldc,
    cublasStatus_t *status)
{
    if (!gemm_eligible(m,n,k)) return 0;
    int devices[8]; const int count=devices_for_operation(devices);
    if (count<2) return 0;
    try {
        if (!fast_gemm_topology(devices,count)) return 0;
        Admission admission(devices,count);
        if (!admission.ok) return 0;
        const int handled=openmx_mgpuDgemmTry(handle,count,devices,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,status);
        report_gemm(m,n,k,'D');
        return handled;
    } catch (const std::bad_alloc &) { return 0; }
}

extern "C" int openmx_mgpu_dispatch_zgemm(cublasHandle_t handle,
    cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const cuDoubleComplex *alpha,
    const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,
    const cuDoubleComplex *beta,cuDoubleComplex *C,int ldc,cublasStatus_t *status)
{
    if (!gemm_eligible(m,n,k)) return 0;
    int devices[8]; const int count=devices_for_operation(devices);
    if (count<2) return 0;
    try {
        if (!fast_gemm_topology(devices,count)) return 0;
        Admission admission(devices,count);
        if (!admission.ok) return 0;
        const int handled=openmx_mgpuZgemmTry(handle,count,devices,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,status);
        report_gemm(m,n,k,'Z');
        return handled;
    } catch (const std::bad_alloc &) { return 0; }
}
