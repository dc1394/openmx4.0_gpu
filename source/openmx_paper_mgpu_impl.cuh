/* Shared real/Hermitian implementation; included twice with PAPER_COMPLEX. */
#if PAPER_COMPLEX
namespace paper_complex {
using Scalar = PaperComplex;
constexpr cublasOperation_t ADJOINT = CUBLAS_OP_C;
#else
namespace paper_real {
using Scalar = double;
constexpr cublasOperation_t ADJOINT = CUBLAS_OP_T;
#endif

constexpr int BW = 32;
constexpr int BAND_LD = 2 * BW;

using Clock = std::chrono::steady_clock;

void ck(cudaError_t e, const char *what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
void ck(cublasStatus_t e, const char *what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what)+": cuBLAS "+std::to_string(int(e)));
}
void ck(cusolverStatus_t e, const char *what) {
    if (e != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string(what)+": cuSOLVER "+std::to_string(int(e)));
}
#define CUDA(call) ck((call), #call)
#define BLAS(call) ck((call), #call)
#define SOLVER(call) ck((call), #call)

/* Keep asynchronous staging memory alive through exceptional unwinding.
 * Normal execution already synchronizes at its dependency boundaries. */
struct StreamOnException {
    cudaStream_t stream; int device,level=std::uncaught_exceptions();
    StreamOnException(cudaStream_t s,int d):stream(s),device(d){}
    ~StreamOnException() {
        if(stream && std::uncaught_exceptions()>level) {
            int previous=-1;cudaGetDevice(&previous);cudaSetDevice(device);
            cudaStreamSynchronize(stream);if(previous>=0)cudaSetDevice(previous);
        }
    }
};
void test_failure(const char *stage) {
#ifdef OPENMX_PAPER_TESTING
    const char *request=std::getenv("OPENMX_PAPER_TEST_FAIL");
    if(request && !std::strcmp(request,stage))throw std::runtime_error(std::string("Injected paper failure: ")+stage);
#else
    (void)stage;
#endif
}

template<class T> void alloc(T *&p, size_t count) {
    CUDA(cudaMalloc(reinterpret_cast<void **>(&p), std::max(size_t(1),count)*sizeof(T)));
}


cublasStatus_t gemm(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
    const Scalar *alpha,const Scalar *a,int lda,const Scalar *b,int ldb,const Scalar *beta,Scalar *c,int ldc) {
#if PAPER_COMPLEX
    return cublasZgemm(h,ta,tb,m,n,k,reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(a),lda,reinterpret_cast<const cuDoubleComplex *>(b),ldb,
        reinterpret_cast<const cuDoubleComplex *>(beta),reinterpret_cast<cuDoubleComplex *>(c),ldc);
#else
    return cublasDgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
#endif
}
cublasStatus_t rank2k(cublasHandle_t h,cublasFillMode_t u,cublasOperation_t op,int n,int k,
    const Scalar *alpha,const Scalar *a,int lda,const Scalar *b,int ldb,const Scalar *beta,Scalar *c,int ldc) {
#if PAPER_COMPLEX
    double br=real_value(*beta);
    return cublasZher2k(h,u,op,n,k,reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(a),lda,reinterpret_cast<const cuDoubleComplex *>(b),ldb,
        &br,reinterpret_cast<cuDoubleComplex *>(c),ldc);
#else
    return cublasDsyr2k(h,u,op,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
#endif
}
cusolverStatus_t qr_buffer(cusolverDnHandle_t h,int m,int n,Scalar *a,int lda,int *size) {
#if PAPER_COMPLEX
    return cusolverDnZgeqrf_bufferSize(h,m,n,reinterpret_cast<cuDoubleComplex *>(a),lda,size);
#else
    return cusolverDnDgeqrf_bufferSize(h,m,n,a,lda,size);
#endif
}
cusolverStatus_t qr(cusolverDnHandle_t h,int m,int n,Scalar *a,int lda,Scalar *tau,Scalar *work,int size,int *info) {
#if PAPER_COMPLEX
    return cusolverDnZgeqrf(h,m,n,reinterpret_cast<cuDoubleComplex *>(a),lda,
        reinterpret_cast<cuDoubleComplex *>(tau),reinterpret_cast<cuDoubleComplex *>(work),size,info);
#else
    return cusolverDnDgeqrf(h,m,n,a,lda,tau,work,size,info);
#endif
}
__global__ void phase_rows(int n,int cols,Scalar *q,const Scalar *phase) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(n)*cols)q[i]*=conj_value(phase[i%n]);
}

struct Reflector { int row; double tau; Scalar v[BW]; };
struct BCTask { int row, col, slot, steps; };
struct Panel { int row=0, width=0; std::vector<Scalar> y,w; };

__global__ void make_y(int m,int k,const Scalar *qr,int ld,Scalar *y) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*k) { int r=i%m,c=i/m; y[i]=r<c?0.0:(r==c?1.0:qr[r+size_t(c)*ld]); }
}
/* Forward, columnwise LARFT recurrence. The small triangular factor is
 * constructed on-device from Y^T Y and the GEQRF tau coefficients. */
__global__ void make_t(int k,const Scalar *gram,const Scalar *tau,Scalar *t) {
    if(threadIdx.x || blockIdx.x) return;
    for(int j=0;j<k;++j) {
        for(int i=0;i<k;++i) t[i+k*j]=0.0;
        for(int i=0;i<j;++i) {
            Scalar s=0; for(int l=i;l<j;++l) s+=t[i+k*l]*gram[l+k*j];
            t[i+k*j]=-tau[j]*s;
        }
        t[j+k*j]=tau[j];
    }
}
__global__ void restore_r(int m,int b,Scalar *a,int ld) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*b && i%m>i/m) a[i%m+size_t(i/m)*ld]=0.0;
}
__global__ void mirror_lower(int n,Scalar *a,int ld) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;
    const int c=blockIdx.y*blockDim.y+threadIdx.y;
    if(r<c && c<n) a[r+size_t(c)*ld]=conj_value(a[c+size_t(r)*ld]);
}
__global__ void pack_band(int n,int first,int cols,const Scalar *a,int ld,Scalar *band) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<BAND_LD*cols) {
        int d=i%BAND_LD,c=i/BAND_LD;
        band[i]=(d<=BW && first+c+d<n)?a[first+c+d+size_t(c)*ld]:0.0;
    }
}
__global__ void identity_columns(int n,int first,int cols,Scalar *q) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(n)*cols) q[i]=(int(i%n)==first+int(i/n))?1.0:0.0;
}
__device__ Scalar warp_sum(Scalar x) {
    for(int d=16;d;d/=2) x+=shuffle_xor(x,d);
    return x;
}
__device__ double warp_max(double x) {
    for(int d=16;d;d/=2) x=fmax(x,__shfl_xor_sync(0xffffffff,x,d));
    return x;
}
__device__ Scalar band_get(const Scalar *a,int start,int r,int c,int n) {
    bool upper=r<c;
    if(upper) {int t=r;r=c;c=t;}
    if(r>=n || c<start || r-c>=BAND_LD) return 0.0;
    Scalar value=a[size_t(c-start)*BAND_LD+r-c];
    return upper?conj_value(value):value;
}
__device__ void band_put(Scalar *a,int start,int r,int c,int n,Scalar v) {
    if(r>=n || r<c || c<start || r-c>=BAND_LD) return;
    a[size_t(c-start)*BAND_LD+r-c]=v;
}
/* A warp owns one Householder operation within one bulge sweep. Different
 * blocks in a launch have disjoint update regions, enforced by the host
 * sweep-dependency scheduler. Two tiles are staged; the lower tile is
 * streamed independently. A warp can consume several ready local bulges. */
__global__ void chase_band(int n,int start,Scalar *a,const BCTask *tasks,
                           Reflector *out) {
    const BCTask task=tasks[blockIdx.x];
    const int lane=threadIdx.x;
    __shared__ Scalar left[BW*BW],diag[BW*BW],v[BW],z[BW];
    for(int step=0;step<task.steps;++step) {
    const int q=task.row+step*BW,c0=step? q-BW:task.col,l=min(BW,n-q);
    const int prev=q-c0,next=min(BW,n-q-l);
    Scalar x=lane<l?band_get(a,start,q+lane,c0,n):0;
    double tail=warp_max(lane>0?abs_value(x):0.0);
    double scale=warp_max(abs_value(x));
    double norm=scale?scale*sqrt(real_value(warp_sum(Scalar(abs_squared(x/scale))))):0;
    Scalar x0=shuffle_lane(x,0);
    double ax0=abs_value(x0);
    Scalar alpha=ax0?-norm*(x0/ax0):Scalar(-norm);
    double tau=(tail==0 || l<2)?0:(norm+ax0)/norm;
    v[lane]=lane==0?1.0:((lane<l && tau!=0)?x/(x0-alpha):0.0);
    for(int j=0;j<BW;++j) {
        left[lane+BW*j]=(lane<l && j<prev)?band_get(a,start,q+lane,c0+j,n):0;
        diag[lane+BW*j]=(lane<l && j<l)?band_get(a,start,q+lane,q+j,n):0;
    }
    __syncwarp();
    Scalar av=0;
    for(int j=0;j<l;++j) av+=diag[lane+BW*j]*v[j];
    double vv=real_value(warp_sum(lane<l?conj_value(v[lane])*av:Scalar(0)));
    z[lane]=tau*av-0.5*tau*tau*vv*v[lane];
    __syncwarp();
    if(lane<prev) {
        Scalar dot=0; for(int i=0;i<l;++i) dot+=conj_value(v[i])*left[i+BW*lane];
        for(int i=0;i<l;++i) {
            Scalar val=left[i+BW*lane]-tau*v[i]*dot;
            if(lane==0 && tau!=0) val=i==0?alpha:0.0;
            band_put(a,start,q+i,c0+lane,n,val);
        }
    }
    if(lane<l) for(int j=0;j<=lane;++j)
        band_put(a,start,q+lane,q+j,n,diag[lane+BW*j]-v[lane]*conj_value(z[j])-z[lane]*conj_value(v[j]));
    if(lane<next) {
        Scalar dot=0; for(int j=0;j<l;++j) dot+=band_get(a,start,q+l+lane,q+j,n)*v[j];
        for(int j=0;j<l;++j)
            band_put(a,start,q+l+lane,q+j,n,band_get(a,start,q+l+lane,q+j,n)-tau*dot*conj_value(v[j]));
    }
    if(lane==0) {out[task.slot+step].row=q;out[task.slot+step].tau=tau;}
    out[task.slot+step].v[lane]=v[lane];
    __syncwarp();
    }
}
/* BLAS2 backtransform: four independent identity columns per block. A group
 * of reflectors is shared across warps, without LARFT/preprocessing. */
__global__ void back_band(int n,int cols,Scalar *q,const Reflector *u,size_t count) {
    constexpr int GROUP=16;
    __shared__ Reflector cache[GROUP];
    const int lane=threadIdx.x,col=blockIdx.x*blockDim.y+threadIdx.y;
    const int t=threadIdx.y*32+lane,nt=blockDim.y*32;
    for(size_t base=0;base<count;base+=GROUP) {
        int num=int(min(size_t(GROUP),count-base));
        const unsigned char *src=reinterpret_cast<const unsigned char *>(u+base);
        unsigned char *dst=reinterpret_cast<unsigned char *>(cache);
        for(int j=t;j<num*int(sizeof(Reflector));j+=nt) dst[j]=src[j];
        __syncthreads();
        for(int j=0;j<num;++j) {
            int row=cache[j].row+lane;
            Scalar x=(col<cols && row<n)?q[row+size_t(col)*n]:0;
            Scalar dot=warp_sum(x*conj_value(cache[j].v[lane]));
            if(col<cols && row<n) q[row+size_t(col)*n]=x-cache[j].tau*cache[j].v[lane]*dot;
        }
        __syncthreads();
    }
}

struct Device {
    int id,start,end,qstart,qcols,n;
    cudaStream_t s=nullptr,bc=nullptr,back=nullptr;
    cublasHandle_t blas=nullptr,backblas=nullptr;
    cusolverDnHandle_t solver=nullptr;
    Scalar *a=nullptr,*y=nullptr,*w=nullptr,*z=nullptr,*gram=nullptr,*t=nullptr;
    Scalar *tau=nullptr,*qrwork=nullptr,*packed=nullptr,*band=nullptr;
    Scalar *qt=nullptr,*by=nullptr,*bw=nullptr,*bt=nullptr;
    int *info=nullptr,qrsize=0;
    BCTask *tasks=nullptr;
    Reflector *u=nullptr;
    std::vector<size_t> uids;
    double first_bc=-1,first_bc_complete=-1,last_bc=-1,back_begin=-1,back_end=-1;
    Device(int d,int lo,int hi,int qlo,int qw,int order):
        id(d),start(lo),end(hi),qstart(qlo),qcols(qw),n(order) {}
    void init(const Scalar *input) {
        CUDA(cudaSetDevice(id));
        CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&bc,cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&back,cudaStreamNonBlocking));
        StreamOnException staging_guard(s,id),band_guard(bc,id);
        BLAS(cublasCreate(&blas));BLAS(cublasSetStream(blas,s));
        BLAS(cublasCreate(&backblas));BLAS(cublasSetStream(backblas,back));
        SOLVER(cusolverDnCreate(&solver));SOLVER(cusolverDnSetStream(solver,s));
        alloc(a,size_t(n)*(end-start));alloc(y,size_t(n)*BW);alloc(w,size_t(n)*BW);
        alloc(z,size_t(n)*BW);alloc(gram,BW*BW);alloc(t,BW*BW);alloc(tau,BW);alloc(info,1);
        alloc(packed,BAND_LD*BW);alloc(band,size_t(BAND_LD)*(end-start+2*BW));
        alloc(tasks,n);alloc(qt,size_t(n)*qcols);alloc(by,size_t(n)*BW);
        alloc(bw,size_t(n)*BW);alloc(bt,size_t(BW)*qcols);
        size_t uc=0;
        for(int sweep=0;sweep<n-2;++sweep) for(int row=sweep+1;row<n-1;row+=BW) {
            int col=row==sweep+1?sweep:row-BW;
            if(col>=start && col<end) ++uc;
        }
        alloc(u,uc);uids.reserve(uc);
        SOLVER(qr_buffer(solver,n,BW,a,n,&qrsize));alloc(qrwork,qrsize);
        CUDA(cudaMemcpyAsync(a,input+size_t(start)*n,size_t(n)*(end-start)*sizeof(Scalar),cudaMemcpyHostToDevice,s));
        CUDA(cudaMemsetAsync(band,0,size_t(BAND_LD)*(end-start+2*BW)*sizeof(Scalar),bc));
        CUDA(cudaStreamSynchronize(s));CUDA(cudaStreamSynchronize(bc));
    }
    ~Device() {
        cudaSetDevice(id);
        if(s) cudaStreamSynchronize(s);if(bc) cudaStreamSynchronize(bc);if(back) cudaStreamSynchronize(back);
        for(Scalar *p:{a,y,w,z,gram,t,tau,qrwork,packed,band,qt,by,bw,bt}) if(p) cudaFree(p);
        if(info)cudaFree(info);if(tasks)cudaFree(tasks);if(u)cudaFree(u);
        if(solver)cusolverDnDestroy(solver);if(blas)cublasDestroy(blas);if(backblas)cublasDestroy(backblas);
        if(s)cudaStreamDestroy(s);if(bc)cudaStreamDestroy(bc);if(back)cudaStreamDestroy(back);
    }
};

struct DeviceStreamsOnException {
    std::vector<std::unique_ptr<Device>> &devices;
    bool band; int level=std::uncaught_exceptions();
    DeviceStreamsOnException(std::vector<std::unique_ptr<Device>> &d,bool b):devices(d),band(b){}
    ~DeviceStreamsOnException() {
        if(std::uncaught_exceptions()>level) {
            int previous=-1;cudaGetDevice(&previous);
            for(auto &d:devices){cudaSetDevice(d->id);cudaStreamSynchronize(band?d->bc:d->s);}
            if(previous>=0)cudaSetDevice(previous);
        }
    }
};

struct Pipeline {
    int n,ng,nev; bool profile,bc_batch;
    size_t bc_launches=0,bc_rounds=0;
    std::vector<std::unique_ptr<Device>> dev;
    std::vector<Panel> panels;
    std::vector<Scalar> hostband,hostdiag,hostsub,phase;
    std::vector<double> d,e,qd;
    std::vector<Reflector> reflectors;
    std::vector<size_t> offsets;
    std::vector<bool> slab_done;
    std::mutex mutex;std::condition_variable cv;
    int ready_cols=0,ready_panels=0;bool bc_done=false,sbr_done=false,abort=false;
    std::string failure;
    Clock::time_point origin=Clock::now();
    double sbr_end=0,bc_end=0,dc_begin=0,dc_end=0;
    Pipeline(int order,int g,int wanted):n(order),ng(g),nev(wanted),
      profile(std::getenv("OPENMX_PAPER_MGPU_PROFILE") && std::atoi(std::getenv("OPENMX_PAPER_MGPU_PROFILE"))!=0),
      bc_batch(std::getenv("OPENMX_PAPER_MGPU_BC_BATCH") && std::atoi(std::getenv("OPENMX_PAPER_MGPU_BC_BATCH"))!=0),
      panels((n+BW-1)/BW),hostband(size_t(BAND_LD)*n),hostdiag(n),hostsub(n),phase(n,Scalar(1)),d(n),e(n),qd(size_t(n)*n),
      offsets(n),slab_done(g,false) {
        size_t count=0;
        for(int s=0;s<n-2;++s) {offsets[s]=count;count+=size_t((n-3-s)/BW+1);}
        reflectors.resize(count);
    }
    ~Pipeline() {
        /* Members holding host staging arrays are destroyed before dev;
         * drain here, while those arrays are still alive. All workers have
         * been joined before a Pipeline leaves the solve's scope. */
        for(auto &x:dev) {
            cudaSetDevice(x->id);
            if(x->s)cudaStreamSynchronize(x->s);
            if(x->bc)cudaStreamSynchronize(x->bc);
            if(x->back)cudaStreamSynchronize(x->back);
        }
    }
    double now()const {return std::chrono::duration<double,std::milli>(Clock::now()-origin).count();}
    void fail(const std::string &s) {
        std::lock_guard<std::mutex>lock(mutex);if(!abort)failure=s;abort=true;cv.notify_all();
    }
    template<class F> void guarded(F fn) {
        try{fn();}catch(const std::exception &x){fail(x.what());}catch(...){fail("Unknown worker failure");}
    }
    bool wait_for_slab(int g) {
        std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return abort||slab_done[g];});return !abort;
    }
    void publish(int j,int cols) {
        std::lock_guard<std::mutex>lock(mutex);ready_cols=j+cols;++ready_panels;
        for(int g=0;g<ng;++g) if(ready_cols>=dev[g]->end)slab_done[g]=true;
        cv.notify_all();
    }
    void sbr() {
        const Scalar one=1,zero=0,neg=-1,half=-.5;
        for(int j=0;j<n;j+=BW) {
            {std::lock_guard<std::mutex>lock(mutex);if(abort)return;}
            int owner=0;while(j>=dev[owner]->end)++owner;
            Device &p=*dev[owner];CUDA(cudaSetDevice(p.id));
            const int cols=std::min(BW,n-j),q=j+BW,m=n-q,k=std::min(BW,m);
            Panel &panel=panels[j/BW];
            if(m>1) {
                panel.row=q;panel.width=k;
                Scalar *ap=p.a+q+size_t(j-p.start)*n;
                SOLVER(qr(p.solver,m,cols,ap,n,p.tau,p.qrwork,p.qrsize,p.info));
                make_y<<<(m*k+255)/256,256,0,p.s>>>(m,k,ap,n,p.y);
                BLAS(gemm(p.blas,ADJOINT,CUBLAS_OP_N,k,k,m,&one,p.y,m,p.y,m,&zero,p.gram,k));
                make_t<<<1,1,0,p.s>>>(k,p.gram,p.tau,p.t);
                BLAS(gemm(p.blas,CUBLAS_OP_N,CUBLAS_OP_N,m,k,k,&one,p.y,m,p.t,k,&zero,p.w,m));
                panel.y.resize(size_t(m)*k);panel.w.resize(size_t(m)*k);
                CUDA(cudaMemcpyAsync(panel.y.data(),p.y,panel.y.size()*sizeof(Scalar),cudaMemcpyDeviceToHost,p.s));
                CUDA(cudaMemcpyAsync(panel.w.data(),p.w,panel.w.size()*sizeof(Scalar),cudaMemcpyDeviceToHost,p.s));
                int info=0;StreamOnException info_guard(p.s,p.id);
                CUDA(cudaMemcpyAsync(&info,p.info,sizeof(info),cudaMemcpyDeviceToHost,p.s));
                restore_r<<<(m*cols+255)/256,256,0,p.s>>>(m,cols,ap,n);
                CUDA(cudaStreamSynchronize(p.s));if(info)throw std::runtime_error("Panel GEQRF failed: "+std::to_string(info));
                std::vector<Scalar> hz(size_t(m)*k);
                DeviceStreamsOnException z_guard(dev,false);
                for(int g=owner;g<ng;++g) {
                    Device &x=*dev[g];int c=std::max(q,x.start),nc=x.end-c;if(nc<=0)continue;
                    CUDA(cudaSetDevice(x.id));
                    if(g!=owner) {
                        CUDA(cudaMemcpyAsync(x.y,panel.y.data(),panel.y.size()*sizeof(Scalar),cudaMemcpyHostToDevice,x.s));
                        CUDA(cudaMemcpyAsync(x.w,panel.w.data(),panel.w.size()*sizeof(Scalar),cudaMemcpyHostToDevice,x.s));
                    }
                    BLAS(gemm(x.blas,ADJOINT,CUBLAS_OP_N,nc,k,m,&one,
                        x.a+q+size_t(c-x.start)*n,n,x.w,m,&zero,x.z+(c-q),m));
                    CUDA(cudaMemcpy2DAsync(hz.data()+(c-q),size_t(m)*sizeof(Scalar),x.z+(c-q),size_t(m)*sizeof(Scalar),
                        size_t(nc)*sizeof(Scalar),k,cudaMemcpyDeviceToHost,x.s));
                }
                test_failure("sbr");
                for(int g=owner;g<ng;++g){CUDA(cudaSetDevice(g));CUDA(cudaStreamSynchronize(dev[g]->s));}
                CUDA(cudaSetDevice(p.id));
                CUDA(cudaMemcpyAsync(p.z,hz.data(),hz.size()*sizeof(Scalar),cudaMemcpyHostToDevice,p.s));
                BLAS(gemm(p.blas,ADJOINT,CUBLAS_OP_N,k,k,m,&one,p.w,m,p.z,m,&zero,p.gram,k));
                BLAS(gemm(p.blas,CUBLAS_OP_N,CUBLAS_OP_N,m,k,k,&half,p.y,m,p.gram,k,&one,p.z,m));
                CUDA(cudaMemcpyAsync(hz.data(),p.z,hz.size()*sizeof(Scalar),cudaMemcpyDeviceToHost,p.s));
                CUDA(cudaStreamSynchronize(p.s));
                for(int g=owner;g<ng;++g) {
                    Device &x=*dev[g];int c=std::max(q,x.start),nc=x.end-c;if(nc<=0)continue;
                    CUDA(cudaSetDevice(x.id));
                    if(g!=owner)CUDA(cudaMemcpyAsync(x.z,hz.data(),hz.size()*sizeof(Scalar),cudaMemcpyHostToDevice,x.s));
                    Scalar *ac=x.a+q+size_t(c-x.start)*n;
                    int rows=(g==ng-1)?c-q:m;
                    if(rows>0) {
                        BLAS(gemm(x.blas,CUBLAS_OP_N,ADJOINT,rows,nc,k,&neg,x.y,m,x.z+(c-q),m,&one,ac,n));
                        BLAS(gemm(x.blas,CUBLAS_OP_N,ADJOINT,rows,nc,k,&neg,x.z,m,x.y+(c-q),m,&one,ac,n));
                    }
                    if(g==ng-1) {
                        Scalar *square=ac+c-q;
                        BLAS(rank2k(x.blas,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,nc,k,&neg,x.y+(c-q),m,x.z+(c-q),m,&one,square,n));
                        mirror_lower<<<dim3((nc+15)/16,(nc+15)/16),dim3(16,16),0,x.s>>>(nc,square,n);
                    }
                }
                for(int g=owner;g<ng;++g){CUDA(cudaSetDevice(g));CUDA(cudaStreamSynchronize(dev[g]->s));}
            }
            CUDA(cudaSetDevice(p.id));
            pack_band<<<(BAND_LD*cols+255)/256,256,0,p.s>>>(n,j,cols,p.a+size_t(j-p.start)*n,n,p.packed);
            CUDA(cudaMemcpyAsync(hostband.data()+size_t(j)*BAND_LD,p.packed,size_t(cols)*BAND_LD*sizeof(Scalar),cudaMemcpyDeviceToHost,p.s));
            CUDA(cudaStreamSynchronize(p.s));publish(j,cols);
        }
        std::lock_guard<std::mutex>lock(mutex);sbr_done=true;sbr_end=now();cv.notify_all();
    }
    void bc_worker() {
        std::vector<int> next(n),uploaded(ng,0);
        for(int s=0;s<n-2;++s)next[s]=s+1;
        size_t completed=0;
        while(completed<reflectors.size()) {
            int ready;std::vector<bool> freed;
            {std::lock_guard<std::mutex>lock(mutex);if(abort)return;ready=ready_cols;freed=slab_done;}
            for(int g=0;g<ng;++g) {
                Device &x=*dev[g];int count=std::max(0,std::min(ready,x.end)-x.start);
                if(count>uploaded[g]) {
                    CUDA(cudaSetDevice(g));
                    CUDA(cudaMemcpyAsync(x.band+size_t(uploaded[g])*BAND_LD,
                        hostband.data()+size_t(x.start+uploaded[g])*BAND_LD,size_t(count-uploaded[g])*BAND_LD*sizeof(Scalar),cudaMemcpyHostToDevice,x.bc));
                    CUDA(cudaStreamSynchronize(x.bc));uploaded[g]=count;
                }
            }
            std::vector<std::vector<BCTask>> batch(ng);
            std::vector<std::vector<int>> sweeps(ng);
            std::vector<bool> cross(ng,false);
            for(int s=0;s<n-2;++s) {
                int row=next[s];if(row>=n-1)continue;
                if(s && next[s-1]<n-1 && next[s-1]<=row+2*BW)continue;
                if(std::min(n,row+BW)>ready)continue;
                int col=row==s+1?s:row-BW,g=0;while(col>=dev[g]->end)++g;
                if(!freed[g])continue;
                int steps=1;
                if(bc_batch)for(int candidate=row+BW;candidate<n-1;candidate+=BW) {
                    if(s && next[s-1]<n-1 && next[s-1]<=candidate+2*BW)break;
                    if(std::min(n,candidate+BW)>ready)break;
                    if(candidate-BW>=dev[g]->end)break;
                    ++steps;
                }
                batch[g].push_back({row,col,0,steps});sweeps[g].push_back(s);
                int last=row+(steps-1)*BW;
                if(last+std::min(BW,n-last)>dev[g]->end)cross[g]=true;
            }
            /* A crossing task owns its neighbor's halo for this batch. Do not
             * launch any task on that neighbor until the updated halo returns. */
            for(int g=0;g<ng-1;++g)if(cross[g] && !batch[g].empty()) {
                batch[g+1].clear();sweeps[g+1].clear();cross[g+1]=false;
            }
            bool any=false;
            DeviceStreamsOnException task_guard(dev,true);
            for(int g=0;g<ng;++g)if(!batch[g].empty()) {
                any=true;Device &x=*dev[g];CUDA(cudaSetDevice(g));
                if(cross[g]) {
                    int halo=std::min(2*BW,n-x.end);
                    CUDA(cudaMemcpyPeerAsync(x.band+size_t(x.end-x.start)*BAND_LD,g,dev[g+1]->band,g+1,
                        size_t(halo)*BAND_LD*sizeof(Scalar),x.bc));
                }
                for(size_t i=0;i<batch[g].size();++i) {
                    int s=sweeps[g][i];batch[g][i].slot=int(x.uids.size());
                    for(int step=0;step<batch[g][i].steps;++step)
                        x.uids.push_back(offsets[s]+size_t((batch[g][i].row-s-1)/BW+step));
                }
                CUDA(cudaMemcpyAsync(x.tasks,batch[g].data(),batch[g].size()*sizeof(BCTask),cudaMemcpyHostToDevice,x.bc));
                if(x.first_bc<0)x.first_bc=now();
                ++bc_launches;
                chase_band<<<int(batch[g].size()),32,0,x.bc>>>(n,x.start,x.band,x.tasks,x.u);
                CUDA(cudaGetLastError());
            }
            if(any)test_failure("bc");
            if(!any) {
                std::unique_lock<std::mutex>lock(mutex);
                if(sbr_done && ready==n)throw std::runtime_error("BC dependency scheduler made no progress");
                cv.wait_for(lock,std::chrono::milliseconds(1));continue;
            }
            ++bc_rounds;
            for(int g=0;g<ng;++g)if(!batch[g].empty()) {
                CUDA(cudaSetDevice(g));CUDA(cudaStreamSynchronize(dev[g]->bc));dev[g]->last_bc=now();
                if(dev[g]->first_bc_complete<0)dev[g]->first_bc_complete=dev[g]->last_bc;
            }
            for(int g=0;g<ng-1;++g)if(cross[g] && !batch[g].empty()) {
                Device &x=*dev[g];int halo=std::min(2*BW,n-x.end);CUDA(cudaSetDevice(g+1));
                CUDA(cudaMemcpyPeerAsync(dev[g+1]->band,g+1,x.band+size_t(x.end-x.start)*BAND_LD,g,
                    size_t(halo)*BAND_LD*sizeof(Scalar),dev[g+1]->bc));CUDA(cudaStreamSynchronize(dev[g+1]->bc));
            }
            for(int g=0;g<ng;++g) for(size_t i=0;i<batch[g].size();++i) {next[sweeps[g][i]]+=batch[g][i].steps*BW;completed+=batch[g][i].steps;}
        }
        /* n=1/2 has no BC tasks: ensure its band was published and uploaded. */
        {std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return abort||sbr_done;});if(abort)return;}
        for(int g=0;g<ng;++g) {
            Device &x=*dev[g];CUDA(cudaSetDevice(g));
            int width=x.end-x.start;
            if(uploaded[g]<width)CUDA(cudaMemcpyAsync(x.band+size_t(uploaded[g])*BAND_LD,
                hostband.data()+size_t(x.start+uploaded[g])*BAND_LD,size_t(width-uploaded[g])*BAND_LD*sizeof(Scalar),cudaMemcpyHostToDevice,x.bc));
            CUDA(cudaMemcpy2DAsync(hostdiag.data()+x.start,sizeof(Scalar),x.band,BAND_LD*sizeof(Scalar),sizeof(Scalar),width,cudaMemcpyDeviceToHost,x.bc));
            CUDA(cudaMemcpy2DAsync(hostsub.data()+x.start,sizeof(Scalar),x.band+1,BAND_LD*sizeof(Scalar),sizeof(Scalar),width,cudaMemcpyDeviceToHost,x.bc));
            std::vector<Reflector> hu(x.uids.size());
            StreamOnException u_guard(x.bc,g);
            if(!hu.empty())CUDA(cudaMemcpyAsync(hu.data(),x.u,hu.size()*sizeof(Reflector),cudaMemcpyDeviceToHost,x.bc));
            CUDA(cudaStreamSynchronize(x.bc));
            for(size_t i=0;i<hu.size();++i)reflectors[x.uids[i]]=hu[i];
        }
        for(int i=0;i<n;++i) {
            d[i]=real_value(hostdiag[i]);
#if PAPER_COMPLEX
            e[i]=abs_value(hostsub[i]);
            if(i+1<n) {
                phase[i+1]=e[i]?phase[i]*(hostsub[i]/e[i]):Scalar(1);
                phase[i+1]=phase[i+1]/abs_value(phase[i+1]);
            }
#else
            e[i]=hostsub[i];
#endif
        }
        std::lock_guard<std::mutex>lock(mutex);bc_done=true;bc_end=now();cv.notify_all();
    }
    void back_worker(int g) {
        if(!wait_for_slab(g))return;
        Device &x=*dev[g];CUDA(cudaSetDevice(g));x.back_begin=now();
        identity_columns<<<int((size_t(n)*x.qcols+255)/256),256,0,x.back>>>(n,x.qstart,x.qcols,x.qt);
        const Scalar one=1,zero=0,neg=-1;
        for(size_t i=0;i<panels.size();++i) {
            {std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return abort||int(i)<ready_panels;});if(abort)return;}
            const Panel &p=panels[i];if(!p.width)continue;
            int m=n-p.row,k=p.width;
            CUDA(cudaMemcpyAsync(x.by,p.y.data(),p.y.size()*sizeof(Scalar),cudaMemcpyHostToDevice,x.back));
            CUDA(cudaMemcpyAsync(x.bw,p.w.data(),p.w.size()*sizeof(Scalar),cudaMemcpyHostToDevice,x.back));
            BLAS(gemm(x.backblas,ADJOINT,CUBLAS_OP_N,k,x.qcols,m,&one,x.bw,m,x.qt+p.row,n,&zero,x.bt,k));
            BLAS(gemm(x.backblas,CUBLAS_OP_N,CUBLAS_OP_N,m,x.qcols,k,&neg,x.by,m,x.bt,k,&one,x.qt+p.row,n));
        }
        {std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return abort||bc_done;});if(abort)return;}
        Reflector *all=nullptr;
        try {
            alloc(all,reflectors.size());
            if(!reflectors.empty()) {
                CUDA(cudaMemcpyAsync(all,reflectors.data(),reflectors.size()*sizeof(Reflector),cudaMemcpyHostToDevice,x.back));
                back_band<<<(x.qcols+3)/4,dim3(32,4),0,x.back>>>(n,x.qcols,x.qt,all,reflectors.size());
                CUDA(cudaGetLastError());
            }
#if PAPER_COMPLEX
            CUDA(cudaMemcpyAsync(x.by,phase.data(),size_t(n)*sizeof(Scalar),cudaMemcpyHostToDevice,x.back));
            phase_rows<<<int((size_t(n)*x.qcols+255)/256),256,0,x.back>>>(n,x.qcols,x.qt,x.by);
#endif
            test_failure("back");
            CUDA(cudaStreamSynchronize(x.back));cudaFree(all);all=nullptr;x.back_end=now();
        } catch(...) {cudaStreamSynchronize(x.back);if(all)cudaFree(all);throw;}
    }
    void dc_worker() {
        {std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return abort||bc_done;});if(abort)return;}
        dc_begin=now();char comp='I';int info=0,lw=-1,li=-1,iq=0;double wq=0;
        dstedc_(&comp,&n,d.data(),e.data(),qd.data(),&n,&wq,&lw,&iq,&li,&info);
        if(info || !std::isfinite(wq) || wq>std::numeric_limits<int>::max())throw std::runtime_error("DSTEDC workspace query failed");
        lw=std::max(1,int(wq));li=std::max(1,iq);std::vector<double>work(lw);std::vector<int>iwork(li);
        dstedc_(&comp,&n,d.data(),e.data(),qd.data(),&n,work.data(),&lw,iwork.data(),&li,&info);
        if(info)throw std::runtime_error("DSTEDC failed: "+std::to_string(info));dc_end=now();
    }
    void finish_one(int g,std::vector<Scalar> &out) {
        const Scalar one=1,zero=0;constexpr int TILE=256;
        {
            Device &x=*dev[g];CUDA(cudaSetDevice(g));Scalar *zd=nullptr,*result=nullptr;
            try {
                alloc(zd,size_t(n)*std::min(TILE,nev));alloc(result,size_t(x.qcols)*std::min(TILE,nev));
                std::vector<Scalar> hostz(size_t(n)*std::min(TILE,nev));
                StreamOnException qd_guard(x.back,g);
                for(int c=0;c<nev;c+=TILE) {
                    int width=std::min(TILE,nev-c);
                    for(size_t i=0;i<size_t(n)*width;++i)hostz[i]=Scalar(qd[size_t(c)*n+i]);
                    CUDA(cudaMemcpyAsync(zd,hostz.data(),size_t(n)*width*sizeof(Scalar),cudaMemcpyHostToDevice,x.back));
                    BLAS(gemm(x.backblas,ADJOINT,CUBLAS_OP_N,x.qcols,width,n,&one,x.qt,n,zd,n,&zero,result,x.qcols));
                    CUDA(cudaMemcpy2DAsync(out.data()+x.qstart+size_t(c)*n,size_t(n)*sizeof(Scalar),result,size_t(x.qcols)*sizeof(Scalar),
                        size_t(x.qcols)*sizeof(Scalar),width,cudaMemcpyDeviceToHost,x.back));
                    test_failure("final");
                    CUDA(cudaStreamSynchronize(x.back));
                }
                cudaFree(zd);zd=nullptr;cudaFree(result);result=nullptr;
            }catch(...){if(zd)cudaFree(zd);if(result)cudaFree(result);throw;}
        }
    }
    void finish(std::vector<Scalar> &out) {
        std::vector<std::thread> jobs;
        try {
            for(int g=0;g<ng;++g)jobs.emplace_back([&,g]{guarded([&]{finish_one(g,out);});});
        } catch(const std::exception &e){fail(e.what());}
        for(auto &thread:jobs)if(thread.joinable())thread.join();
        if(abort)throw std::runtime_error(failure);
    }
    void report() {
        if(!profile)return;
        std::fprintf(stderr,"PAPER_MGPU type=%s n=%d nev=%d ngpu=%d bw=%d sbr_end_ms=%.6f bc_end_ms=%.6f dc_start_ms=%.6f dc_end_ms=%.6f total_ms=%.6f\n",
            PAPER_COMPLEX?"Z":"D",n,nev,ng,BW,sbr_end,bc_end,dc_begin,dc_end,now());
        std::fprintf(stderr,"PAPER_MGPU_BC batch=%d rounds=%zu launches=%zu reflectors=%zu\n",
            int(bc_batch),bc_rounds,bc_launches,reflectors.size());
        for(int g=0;g<ng;++g) {
            const Device &x=*dev[g];
            std::fprintf(stderr,"PAPER_MGPU_GPU gpu=%d cols=%d:%d bc_first_ms=%.6f bc_first_complete_ms=%.6f bc_last_ms=%.6f back_start_ms=%.6f back_end_ms=%.6f early_bc=%d\n",
                g,x.start,x.end,x.first_bc,x.first_bc_complete,x.last_bc,x.back_begin,x.back_end,int(x.first_bc_complete>=0 && x.first_bc_complete<sbr_end));
        }
    }
};
int solve(int ngpu,int n,int nev,void *matrix,int lda,double *w) {
    auto *a=static_cast<unsigned char *>(matrix);
    paper_last_error.clear();int original=-1;cudaGetDevice(&original);
    try {
        if(n<1 || nev<1 || nev>n || lda<n || !a || !w || ngpu<1)
            throw std::runtime_error("Invalid paper multi-GPU eigensolver argument");
        if(ngpu>openmx_paper_mgpu_available())throw std::runtime_error("Requested CUDA devices are not visible");
        if(size_t(lda)>std::numeric_limits<size_t>::max()/size_t(n)/sizeof(Scalar))throw std::runtime_error("Matrix size overflow");
        /* Each nonfinal BC slab must contain its 2b halo footprint. */
        int blocks=(n+BW-1)/BW,used=std::min(ngpu,std::max(1,blocks/2));
        Pipeline p(n,used,nev);std::vector<Scalar>input(size_t(n)*n);
        for(int c=0;c<n;++c)for(int r=0;r<n;++r) {
            /* The C ABI accepts interleaved doubles with double alignment;
             * CUDA complex storage has a stronger 16-byte alignment. */
            Scalar val;size_t index=r>=c?r+size_t(c)*lda:c+size_t(r)*lda;
            std::memcpy(&val,a+index*sizeof(Scalar),sizeof(Scalar));
            if(r<c)val=conj_value(val);
            if(r==c)val=Scalar(real_value(val));
            if(!finite_value(val))throw std::runtime_error("Non-finite symmetric matrix input");
            input[r+size_t(c)*n]=val;
        }
        for(int g=0;g<used;++g) {
            int lo=std::min(n,(g*blocks/used)*BW),hi=std::min(n,((g+1)*blocks/used)*BW);
            int qlo=g*n/used,qhi=(g+1)*n/used;
            p.dev.emplace_back(new Device(g,lo,hi,qlo,qhi-qlo,n));p.dev.back()->init(input.data());
        }
        std::vector<std::thread>workers;
        try {
            workers.emplace_back([&]{p.guarded([&]{p.bc_worker();});});
            test_failure("thread");
            workers.emplace_back([&]{p.guarded([&]{p.dc_worker();});});
            for(int g=0;g<used;++g)workers.emplace_back([&,g]{p.guarded([&]{p.back_worker(g);});});
            p.guarded([&]{p.sbr();});
        }catch(const std::exception &e){p.fail(e.what());}
        for(auto &thread:workers)if(thread.joinable())thread.join();
        if(p.abort)throw std::runtime_error(p.failure);
        std::vector<Scalar>out(size_t(n)*nev);p.finish(out);p.report();
        for(double val:p.d)if(!finite_value(val))throw std::runtime_error("Non-finite eigenvalue output");
        for(Scalar val:out)if(!finite_value(val))throw std::runtime_error("Non-finite eigenvector output");
        for(int c=0;c<nev;++c)std::memcpy(a+size_t(c)*lda*sizeof(Scalar),out.data()+size_t(c)*n,size_t(n)*sizeof(Scalar));
        std::memcpy(w,p.d.data(),size_t(nev)*sizeof(double));
    }catch(const std::exception &e){paper_last_error=e.what();if(original>=0)cudaSetDevice(original);return -1;}
    catch(...){paper_last_error="Unknown paper multi-GPU eigensolver error";if(original>=0)cudaSetDevice(original);return -1;}
    if(original>=0)cudaSetDevice(original);return 0;
}

} // namespace selected by PAPER_COMPLEX
