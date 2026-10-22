/* Included inside the MAGMA adapter's private namespace (the includer
   provides openmx_env.h for the ::openmx_env_long_at settings reads). Host-backed
   Ogita-Aishima correction: MAGMA FP32 supplies a complete basis; bounded
   multi-GPU GEMMs form the FP64 corrections. An optional bounded resident
   workspace retains the basis and partitions output columns across GPUs.
   Cluster Rayleigh-Ritz resolves nearly degenerate vectors.
   Reference: Ogita & Aishima, JJIAM 36 (2019), 435-459,
   https://doi.org/10.1007/s13160-019-00348-4 . */

extern "C" {
void dsygv_(const int*,const char*,const char*,const int*,double*,const int*,
            double*,const int*,double*,double*,const int*,int*);
void zhegv_(const int*,const char*,const char*,const int*,void*,const int*,
            void*,const int*,double*,void*,const int*,double*,int*);
void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,
            const double*,const int*,const double*,const int*,const double*,double*,const int*);
void zgemm_(const char*,const char*,const int*,const int*,const int*,const void*,
            const void*,const int*,const void*,const int*,const void*,void*,const int*);
}

template<class T> struct LowPrecision { using Type=float; };
template<> struct LowPrecision<Complex> { using Type=std::complex<float>; };

template<class T>
int refine_product(int ngpu,const int *devices,cublasOperation_t ta,
                   int m,int n,int k,const T *a,int lda,const T *b,int ldb,T *c,int ldc)
{
    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;
    int handled;
    if (sizeof(T)==8) {
        const double one=1,zero=0;
        handled=openmx_mgpuDgemmHostTry(ngpu,devices,ta,CUBLAS_OP_N,m,n,k,&one,
            reinterpret_cast<const double*>(a),lda,reinterpret_cast<const double*>(b),ldb,
            &zero,reinterpret_cast<double*>(c),ldc,&status);
    } else {
        const cuDoubleComplex one={1,0},zero={0,0};
        handled=openmx_mgpuZgemmHostTry(ngpu,devices,ta,CUBLAS_OP_N,m,n,k,&one,
            reinterpret_cast<const cuDoubleComplex*>(a),lda,reinterpret_cast<const cuDoubleComplex*>(b),ldb,
            &zero,reinterpret_cast<cuDoubleComplex*>(c),ldc,&status);
    }
    if (handled && status==CUBLAS_STATUS_SUCCESS) return 0;
    return fail(handled?OPENMX_MAGMA_CUDA_ERROR:OPENMX_MAGMA_RESOURCES,
                "bounded FP64 refinement product failed");
}

template<class T>
int fp32_basis(int ngpu,int n,const std::vector<T> &a,double scale,
               std::vector<T> &x,std::vector<double> &w)
{
    using F=typename LowPrecision<T>::Type;
    const bool complex=sizeof(T)!=8;
    int info=0,found=n==1?1:0,iq=0;
    float rq=0,sq=0;
    magmaFloatComplex cq{};
    int rc;
    if (complex) {
        rc=magma_cheevdx_m(ngpu,MagmaVec,MagmaRangeAll,MagmaLower,n,nullptr,n,0,0,1,n,
                           &found,nullptr,&cq,-1,&rq,-1,&iq,-1,&info);
        sq=MAGMA_C_REAL(cq);
    } else {
        rc=magma_ssyevdx_m(ngpu,MagmaVec,MagmaRangeAll,MagmaLower,n,nullptr,n,0,0,1,n,
                           &found,nullptr,&sq,-1,&iq,-1,&info);
    }
    /* Single precision workspace queries can round an integer down. Add one
       float ulp before conversion, bounded by the LP64 interface. */
    const double lq=std::nextafter(sq,std::numeric_limits<float>::infinity());
    const double lrq=complex?std::nextafter(rq,std::numeric_limits<float>::infinity()):0;
    if (rc || info || !std::isfinite(lq) || lq<1 || lq>INT_MAX ||
        !std::isfinite(lrq) || lrq>INT_MAX || (complex && lrq<1) || iq<1)
        return fail(OPENMX_MAGMA_RESOURCES,"FP32 MAGMA workspace query failed");
    const int lw=int(std::ceil(lq)),lrw=int(std::ceil(lrq));
    std::vector<F> af(size_t(n)*n),work(lw);
    std::vector<float> eigenvalues(n),rwork(lrw);
    std::vector<int> iwork(iq);
    #pragma omp parallel for schedule(static)
    for (size_t i=0;i<af.size();++i) af[i]=F(a[i]/scale);
    if (complex) {
        rc=magma_cheevdx_m(ngpu,MagmaVec,MagmaRangeAll,MagmaLower,n,
            reinterpret_cast<magmaFloatComplex*>(af.data()),n,0,0,1,n,&found,eigenvalues.data(),
            reinterpret_cast<magmaFloatComplex*>(work.data()),lw,rwork.data(),lrw,iwork.data(),iq,&info);
    } else {
        rc=magma_ssyevdx_m(ngpu,MagmaVec,MagmaRangeAll,MagmaLower,n,
            reinterpret_cast<float*>(af.data()),n,0,0,1,n,&found,eigenvalues.data(),
            reinterpret_cast<float*>(work.data()),lw,iwork.data(),iq,&info);
    }
    if (rc || info || found!=n) return fail(OPENMX_MAGMA_LIBRARY_ERROR,"FP32 MAGMA solve failed");
    #pragma omp parallel for schedule(static)
    for (size_t i=0;i<af.size();++i) x[i]=T(af[i]);
    for (int j=0;j<n;++j) w[j]=double(eigenvalues[j])*scale;
    order_eigenpairs(n,n,true,x,w);
    return 0;
}

/* Small generalized solves use native FP64 LAPACK. No loss of precision
   is accepted for an unresolved cluster, including a subset boundary. */
template<class T>
int refine_cluster(int ngpu,const int *devices,int n,int begin,int size,int stride,const T *h,const T *g,
                    std::vector<T> &x,std::vector<double> &w)
{
    std::vector<T> hc(size_t(size)*size),gc(hc.size()),tmp(size_t(n)*size);
    for (int j=0;j<size;++j) for (int i=0;i<size;++i) {
        const size_t at=size_t(begin+j)*stride+begin+i;
        hc[size_t(j)*size+i]=h[at]; gc[size_t(j)*size+i]=g[at];
    }
    int type=1,info=0,lwork=-1;
    T query{};
    std::vector<double> rwork(3*size),values(size);
    const int gpu_min=static_cast<int>(std::max(2L,std::min(long(INT_MAX),::openmx_env_long_at(OPENMX_ENV_MAGMA_RR_GPU_MIN,1024))));
    if(size>=gpu_min) {
        /* A dense spectrum can connect thousands of neighboring states.
           Resolve that subspace on the same admitted GPU group instead of
           imposing a CPU-sized cluster limit or dividing a degeneracy. */
        Session cluster_session;
        int rc=device_preflight(ngpu,devices,size,sizeof(T),cluster_session);
        if(rc) return rc;
        int iq=0;
        double rq=0;
        if(sizeof(T)==8) rc=magma_dsygvd_m(ngpu,1,MagmaVec,MagmaLower,size,
            reinterpret_cast<double*>(hc.data()),size,reinterpret_cast<double*>(gc.data()),size,
            values.data(),reinterpret_cast<double*>(&query),-1,&iq,-1,&info);
        else rc=magma_zhegvd_m(ngpu,1,MagmaVec,MagmaLower,size,
            reinterpret_cast<magmaDoubleComplex*>(hc.data()),size,reinterpret_cast<magmaDoubleComplex*>(gc.data()),size,
            values.data(),reinterpret_cast<magmaDoubleComplex*>(&query),-1,&rq,-1,&iq,-1,&info);
        if(rc || info || !finite(query) || real_part(query)<1 || real_part(query)>INT_MAX ||
           iq<1 || !std::isfinite(rq) || rq>INT_MAX || (sizeof(T)!=8 && rq<1))
            return fail(OPENMX_MAGMA_RESOURCES,"GPU Rayleigh-Ritz workspace query failed");
        lwork=int(std::ceil(real_part(query)));
        std::vector<T> work(lwork);
        rwork.resize(int(std::ceil(rq)));
        std::vector<int> iwork(iq);
        if(sizeof(T)==8) rc=magma_dsygvd_m(ngpu,1,MagmaVec,MagmaLower,size,
            reinterpret_cast<double*>(hc.data()),size,reinterpret_cast<double*>(gc.data()),size,
            values.data(),reinterpret_cast<double*>(work.data()),lwork,iwork.data(),iq,&info);
        else rc=magma_zhegvd_m(ngpu,1,MagmaVec,MagmaLower,size,
            reinterpret_cast<magmaDoubleComplex*>(hc.data()),size,reinterpret_cast<magmaDoubleComplex*>(gc.data()),size,
            values.data(),reinterpret_cast<magmaDoubleComplex*>(work.data()),lwork,rwork.data(),int(rwork.size()),iwork.data(),iq,&info);
        for(int j=0;j<ngpu;++j)
            if(cudaSetDevice(devices[j])!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess)
                return fail(OPENMX_MAGMA_CUDA_ERROR,"GPU Rayleigh-Ritz execution failed");
        if(rc || info) return fail(OPENMX_MAGMA_LIBRARY_ERROR,"GPU Rayleigh-Ritz eigensolver failed");
        if(cluster_session.mapped) {
            if(openmx_magma_device_map_v1(0,nullptr))
                return fail(OPENMX_MAGMA_LIBRARY_ERROR,"Rayleigh-Ritz device map cleanup failed");
            cluster_session.mapped=false;
        }
        rc=refine_product(ngpu,devices,CUBLAS_OP_N,n,size,size,
            x.data()+size_t(begin)*n,n,hc.data(),size,tmp.data(),n);
        if(rc) return rc;
        std::fprintf(stderr,"<OpenMX_MAGMA> Rayleigh-Ritz GPU cluster begin=%d size=%d gpus=%d\n",begin,size,ngpu);
    } else {
    if (sizeof(T)==8) dsygv_(&type,"V","L",&size,reinterpret_cast<double*>(hc.data()),&size,
        reinterpret_cast<double*>(gc.data()),&size,values.data(),reinterpret_cast<double*>(&query),&lwork,&info);
    else zhegv_(&type,"V","L",&size,hc.data(),&size,gc.data(),&size,values.data(),&query,&lwork,rwork.data(),&info);
    if (info || !std::isfinite(real_part(query)) || real_part(query)<1 || real_part(query)>INT_MAX)
        return fail(OPENMX_MAGMA_RESOURCES,"CPU Rayleigh-Ritz workspace query failed");
    lwork=int(std::ceil(real_part(query)));
    std::vector<T> work(lwork);
    if (sizeof(T)==8) dsygv_(&type,"V","L",&size,reinterpret_cast<double*>(hc.data()),&size,
        reinterpret_cast<double*>(gc.data()),&size,values.data(),reinterpret_cast<double*>(work.data()),&lwork,&info);
    else zhegv_(&type,"V","L",&size,hc.data(),&size,gc.data(),&size,values.data(),work.data(),&lwork,rwork.data(),&info);
    if (info) return fail(OPENMX_MAGMA_VALIDATION_ERROR,"CPU Rayleigh-Ritz eigensolver failed");
    const T one=T(1),zero=T(0);
    if (sizeof(T)==8) dgemm_("N","N",&n,&size,&size,reinterpret_cast<const double*>(&one),
        reinterpret_cast<const double*>(x.data()+size_t(begin)*n),&n,reinterpret_cast<const double*>(hc.data()),&size,
        reinterpret_cast<const double*>(&zero),reinterpret_cast<double*>(tmp.data()),&n);
    else zgemm_("N","N",&n,&size,&size,&one,x.data()+size_t(begin)*n,&n,hc.data(),&size,&zero,tmp.data(),&n);
    }
    std::copy(tmp.begin(),tmp.end(),x.begin()+size_t(begin)*n);
    std::copy(values.begin(),values.end(),w.begin()+begin);
    return 0;
}

template<class T>
int refined_solve(int ngpu,const int *devices,int lower,int n,int first,int last,void *A,int lda,double *W)
{
    using Clock=std::chrono::steady_clock;
    const auto start=Clock::now();
    Session session;
    int rc=device_preflight(ngpu,devices,n,sizeof(T),session);
    if (rc) return rc;
    real_workspace.clear(); complex_workspace.clear();
    const size_t nn=size_t(n)*n;
    std::vector<T> a(nn),x(nn);
    std::vector<double> w(n);
    double norm=0;
    int valid=1;
    #pragma omp parallel for schedule(static) reduction(max:norm) reduction(min:valid)
    for (int j=0;j<n;++j) {
        double sum=0;
        for (int i=0;i<n;++i) {
            const T value=matrix<T>(A,lda,i,j,lower!=0);
            if (!finite(value)) valid=0;
            a[size_t(j)*n+i]=value; sum+=abs_bound(value);
        }
        norm=std::max(norm,sum);
    }
    if (!valid || !std::isfinite(norm)) return fail(OPENMX_MAGMA_INVALID,"nonfinite input triangle/norm");
    rc=fp32_basis(ngpu,n,a,norm>0?norm:1,x,w);
    for (int j=0;j<ngpu;++j)
        if (cudaSetDevice(devices[j])!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess)
            return fail(OPENMX_MAGMA_CUDA_ERROR,"FP32 MAGMA device execution failed");
    if (rc) return rc;
    /* MAGMA logical device mapping must not affect the CUDA GEMM workers. */
    if (session.mapped) {
        if (openmx_magma_device_map_v1(0,nullptr)) return fail(OPENMX_MAGMA_LIBRARY_ERROR,"device map cleanup failed");
        session.mapped=false;
    }
    const double fp32_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    std::fprintf(stderr,"<OpenMX_MAGMA> FP32 basis n=%d type=%c gpus=%d seconds=%.6f\n",n,sizeof(T)==8?'D':'Z',ngpu,fp32_seconds);
    struct Resident {
        void *p=nullptr;
        ~Resident() { if(p) openmx_mgpuRefineRelease(p); }
    } resident;
    if(openmx_mgpuRefineCreate && openmx_mgpuRefineProject &&
       openmx_mgpuRefineUpdate && openmx_mgpuRefineRelease)
        resident.p=openmx_mgpuRefineCreate(ngpu,devices,n,sizeof(T)!=8);
    if(resident.p)
        std::fprintf(stderr,"<OpenMX_MAGMA> resident correction n=%d gpus=%d type=%c\n",n,ngpu,sizeof(T)==8?'D':'Z');
    std::vector<T> y(resident.p?0:nn),h(nn),g(nn);
    auto project=[&](int end) -> int {
        if(resident.p) {
            if(openmx_mgpuRefineProject(resident.p,a.data(),x.data(),end,h.data(),g.data()))
                return fail(OPENMX_MAGMA_CUDA_ERROR,"resident refinement projection failed");
            return 0;
        }
        int result;
        if((result=refine_product(ngpu,devices,CUBLAS_OP_N,n,end,n,a.data(),n,x.data(),n,y.data(),n)) ||
           (result=refine_product(ngpu,devices,CUBLAS_OP_C,end,end,n,x.data(),n,y.data(),n,h.data(),end)) ||
           (result=refine_product(ngpu,devices,CUBLAS_OP_C,end,end,n,x.data(),n,x.data(),n,g.data(),end))) return result;
        return 0;
    };
    struct StagingScope {
        bool enabled=openmx_mgpuGemmStagingBegin && openmx_mgpuGemmStagingEnd;
        StagingScope() { if(enabled) openmx_mgpuGemmStagingBegin(); }
        ~StagingScope() { if(enabled) openmx_mgpuGemmStagingEnd(); }
    } staging;
    double delta=0,orth=0,off=0;
    std::vector<int> cluster(n,0);
    const double radius=std::max(std::abs(w.front()),std::abs(w.back()));
    const int iterations=static_cast<int>(std::max(2L,std::min(5L,::openmx_env_long_at(OPENMX_ENV_MAGMA_REFINE_STEPS,2))));
    for (int it=0;it<iterations;++it) {
        if ((rc=project(n))) return rc;
        for (int j=0;j<n;++j) {
            const double den=real_part(g[size_t(j)*n+j]);
            if (!(den>0)) return fail(OPENMX_MAGMA_VALIDATION_ERROR,"nonpositive refinement Gram diagonal");
            w[j]=real_part(h[size_t(j)*n+j])/den;
        }
        orth=off=0; valid=1;
        #pragma omp parallel for schedule(static) reduction(max:orth,off) reduction(min:valid)
        for (int j=0;j<n;++j) for (int i=0;i<n;++i) {
            const size_t at=size_t(j)*n+i;
            if (!finite(h[at]) || !finite(g[at])) valid=0;
            orth=std::max(orth,std::abs(g[at]-T(i==j?1:0)));
            if (i!=j) off=std::max(off,std::abs(h[at]));
        }
        if (!valid) return fail(OPENMX_MAGMA_VALIDATION_ERROR,"nonfinite refinement products");
        if(it==0) delta=std::max(2*(off+radius*orth),64*std::numeric_limits<double>::epsilon()*radius);
        if(it==0) {
            /* The global error estimate can miss a strongly mixed pair in a
               dense spectrum. Put pairs requiring a large first rotation in
               the Rayleigh-Ritz clusters instead of dividing by their small
               gap. Keep this threshold fixed for subsequent corrections:
               shrinking it can split a cluster before it has been resolved. */
            double needed=delta;
            #pragma omp parallel for schedule(static) reduction(max:needed)
            for(int j=0;j<n;++j) for(int i=0;i<n;++i) if(i!=j) {
                const size_t at=size_t(j)*n+i;
                const double gap=w[j]-w[i];
                if(std::abs(gap)>delta && std::abs((h[at]-w[j]*g[at])/gap)>0.05)
                    needed=std::max(needed,std::abs(gap));
            }
            if(needed>delta) {
                const double previous=delta;
                delta=std::nextafter(needed,std::numeric_limits<double>::infinity());
                std::fprintf(stderr,"<OpenMX_MAGMA> expanded refinement clusters n=%d delta=%.6e previous=%.6e\n",
                    n,delta,previous);
            }
        }
        if(it==0) {
            /* Connected clusters stay intact through every correction and
               the final generalized solve. Pairwise gap tests alone rotate
               different parts of the same unresolved subspace inconsistently. */
            for(int j=1;j<n;++j)
                cluster[j]=cluster[j-1]+(std::abs(w[j]-w[j-1])>2*delta?1:0);
        }
        double correction=0;
        #pragma omp parallel for schedule(static) reduction(max:correction)
        for (int j=0;j<n;++j) for (int i=0;i<n;++i) {
            const size_t at=size_t(j)*n+i;
            const T r=T(i==j?1:0)-g[at];
            const double gap=w[j]-w[i];
            g[at]=(cluster[i]!=cluster[j] && std::abs(gap)>delta)?(h[at]+w[j]*r)/gap:T(0.5)*r;
            correction=std::max(correction,std::abs(g[at]));
        }
        if (!std::isfinite(correction) || correction>0.1) {
            std::fprintf(stderr,"<OpenMX_MAGMA> refinement rejected n=%d step=%d orth=%.3e offdiag=%.3e correction=%.3e delta=%.3e radius=%.3e\n",
                n,it+1,orth,off,correction,delta,radius);
            return fail(OPENMX_MAGMA_VALIDATION_ERROR,"FP32 basis outside refinement convergence region");
        }
        if(resident.p) {
            if(openmx_mgpuRefineUpdate(resident.p,g.data(),x.data()))
                return fail(OPENMX_MAGMA_CUDA_ERROR,"resident refinement update failed");
        } else {
            if ((rc=refine_product(ngpu,devices,CUBLAS_OP_N,n,n,n,x.data(),n,g.data(),n,y.data(),n))) return rc;
            #pragma omp parallel for schedule(static)
            for (size_t i=0;i<nn;++i) x[i]+=y[i];
        }
        std::fprintf(stderr,"<OpenMX_MAGMA> refinement n=%d step=%d orth=%.3e offdiag=%.3e correction=%.3e delta=%.3e\n",
            n,it+1,orth,off,correction,delta);
    }
    /* Include the entire near-degenerate cluster at the subset boundary. */
    double chain=2*delta;
    int end=last;
    while (end<n && cluster[end]==cluster[end-1]) ++end;
    if ((rc=project(end))) return rc;
    if(resident.p) {
        rc=openmx_mgpuRefineRelease(resident.p);resident.p=nullptr;
        if(rc) return fail(OPENMX_MAGMA_CUDA_ERROR,"resident refinement workspace cleanup failed");
    }
    orth=off=0;valid=1;
    #pragma omp parallel for schedule(static) reduction(max:orth,off) reduction(min:valid)
    for (int j=0;j<end;++j) for (int i=0;i<end;++i) {
        const size_t at=size_t(j)*end+i;
        if(!finite(h[at]) || !finite(g[at])) valid=0;
        orth=std::max(orth,std::abs(g[at]-T(i==j?1:0)));
        if(i!=j) off=std::max(off,std::abs(h[at]));
    }
    if(!valid) return fail(OPENMX_MAGMA_VALIDATION_ERROR,"nonfinite final refinement products");
    std::fprintf(stderr,"<OpenMX_MAGMA> final products n=%d orth=%.3e offdiag=%.3e cluster_gap=%.3e\n",n,orth,off,chain);
    if(end<n && cluster[end]==cluster[end-1])
        return fail(OPENMX_MAGMA_VALIDATION_ERROR,"refinement subset boundary is unresolved");
    int clusters=0,largest=0;
    for (int begin=0;begin<end;) {
        int next=begin+1;
        while (next<end && cluster[next]==cluster[begin]) ++next;
        const int size=next-begin;
        rc=refine_cluster(ngpu,devices,n,begin,size,end,h.data(),g.data(),x,w);
        if(rc) return rc;
        if (size>1) { ++clusters; largest=std::max(largest,size); }
        begin=next;
    }
    order_eigenpairs(n,end,true,x,w);
    const int count=last-first+1;
    if (first>1) {
        std::copy(x.begin()+size_t(first-1)*n,x.begin()+size_t(last)*n,x.begin());
        std::copy(w.begin()+first-1,w.begin()+last,w.begin());
    }
    /* Refinement is experimental arithmetic: unlike a native library solve,
       it must pass every returned residual and every Gram-matrix entry.
       Reuse the projection workspaces and bounded GPU products. This also
       catches errors confined to unsampled, nearly degenerate eigenvectors. */
    if((rc=refine_product(ngpu,devices,CUBLAS_OP_N,n,count,n,a.data(),n,x.data(),n,h.data(),n)) ||
       (rc=refine_product(ngpu,devices,CUBLAS_OP_C,count,count,n,x.data(),n,x.data(),n,g.data(),count))) return rc;
    double residual=0,gram_error=0;
    valid=1;
    #pragma omp parallel for schedule(static) reduction(max:residual) reduction(min:valid)
    for(int j=0;j<count;++j) {
        double r2=0,v2=0;
        for(int i=0;i<n;++i) {
            const size_t at=size_t(j)*n+i;
            const double r=std::abs(h[at]-w[j]*x[at]),v=std::abs(x[at]);
            r2+=r*r;v2+=v*v;
        }
        const double den=(norm+std::abs(w[j]))*std::sqrt(v2);
        const double error=std::sqrt(r2)/std::max(den,std::numeric_limits<double>::min());
        if(!std::isfinite(error) || !std::isfinite(v2) || v2<0.25) valid=0;
        residual=std::max(residual,error);
    }
    #pragma omp parallel for schedule(static) reduction(max:gram_error) reduction(min:valid)
    for(int j=0;j<count;++j) for(int i=0;i<count;++i) {
        const double error=std::abs(g[size_t(j)*count+i]-T(i==j?1:0));
        if(!std::isfinite(error)) valid=0;
        gram_error=std::max(gram_error,error);
    }
    std::fprintf(stderr,"<OpenMX_MAGMA> all-vector refinement validation n=%d nev=%d residual=%.9e orth=%.9e valid=%d\n",
        n,count,residual,gram_error,int(valid && residual<=1e-10 && gram_error<=1e-10));
    if(!valid || residual>1e-10 || gram_error>1e-10)
        return fail(OPENMX_MAGMA_VALIDATION_ERROR,"all-vector refinement residual/orthogonality validation failed");
    if (!validate<T>(A,lda,lower!=0,n,count,x,w,true,norm)) return OPENMX_MAGMA_VALIDATION_ERROR;
    if (cudaSetDevice(session.previous)!=cudaSuccess) return fail(OPENMX_MAGMA_CUDA_ERROR,"restore device after refinement failed");
    for (int j=0;j<count;++j)
        std::memcpy(static_cast<unsigned char*>(A)+size_t(j)*lda*sizeof(T),x.data()+size_t(j)*n,size_t(n)*sizeof(T));
    std::memcpy(W,w.data(),size_t(count)*sizeof(double));
    std::fprintf(stderr,"<OpenMX_MAGMA> FP32 refined n=%d nev=%d type=%c gpus=%d steps=%d clusters=%d largest=%d seconds=%.6f valid=1\n",
        n,count,sizeof(T)==8?'D':'Z',ngpu,iterations,clusters,largest,std::chrono::duration<double>(Clock::now()-start).count());
    return 0;
}
