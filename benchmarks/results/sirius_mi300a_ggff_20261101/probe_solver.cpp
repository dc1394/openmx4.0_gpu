#include <hip/hip_runtime.h>
#include <hipsolver/hipsolver.h>
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP(x) do {auto code=(x);if(code!=hipSuccess){fprintf(stderr,"HIP %s: %d\n",#x,(int)code);MPI_Abort(MPI_COMM_WORLD,2);}}while(0)
#define SOLVER(x) do {auto code=(x);if(code!=HIPSOLVER_STATUS_SUCCESS){fprintf(stderr,"solver %s: %d\n",#x,(int)code);MPI_Abort(MPI_COMM_WORLD,3);}}while(0)
static double residual(const std::vector<hipDoubleComplex>& a,const std::vector<hipDoubleComplex>& v,
                       const std::vector<double>& w,int n,int nev)
{
    long double norm=0;
    for(auto z:a)norm+=(long double)z.x*z.x+(long double)z.y*z.y;
    norm=std::sqrt(norm);
    long double worst=0;
    for(int col: {0,1,nev/3,nev/2,nev-2,nev-1}){
        long double r2=0,v2=0;
        for(int i=0;i<n;i++){
            long double r=0,im=0;
            for(int k=0;k<n;k++){
                auto x=a[(size_t)k*n+i],y=v[(size_t)col*n+k];
                r+=(long double)x.x*y.x-(long double)x.y*y.y;
                im+=(long double)x.x*y.y+(long double)x.y*y.x;
            }
            auto y=v[(size_t)col*n+i];
            r-=(long double)w[col]*y.x;im-=(long double)w[col]*y.y;
            r2+=r*r+im*im;v2+=(long double)y.x*y.x+(long double)y.y*y.y;
        }
        long double value=std::sqrt(r2)/(std::max(norm,(long double)1)*std::sqrt(v2));
        if(!std::isfinite(value))return INFINITY;
        worst=std::max(worst,value);
    }
    return (double)worst;
}
int main(int argc,char**argv)
{
    int provided,rank,ranks,devices;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if(argc<2)MPI_Abort(MPI_COMM_WORLD,1);
    FILE *f=fopen(argv[1],"rb");uint64_t sizes[2];
    if(!f || fread(sizes,sizeof(sizes),1,f)!=1)MPI_Abort(MPI_COMM_WORLD,1);
    int n=(int)sizes[0],nev=(int)sizes[1];
    std::vector<hipDoubleComplex>a((size_t)n*n),v((size_t)n*nev);
    std::vector<double>w(n),reference(n);
    if(fread(a.data(),a.size()*sizeof(a[0]),1,f)!=1)MPI_Abort(MPI_COMM_WORLD,1);fclose(f);
    HIP(hipGetDeviceCount(&devices));
    int block=(ranks+devices-1)/devices;
    HIP(hipSetDevice(std::min(rank/block,devices-1)));
    hipStream_t stream;hipsolverDnHandle_t h;
    HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    SOLVER(hipsolverDnCreate(&h));SOLVER(hipsolverDnSetStream(h,stream));
    hipDoubleComplex *da,*work;double *dw;int *di;
    HIP(hipMalloc(&da,a.size()*sizeof(a[0])));HIP(hipMalloc(&dw,n*sizeof(double)));HIP(hipMalloc(&di,sizeof(int)));
    int full_work=0,partial_work=0,found=0;
    SOLVER(hipsolverDnZheevd_bufferSize(h,HIPSOLVER_EIG_MODE_VECTOR,HIPBLAS_FILL_MODE_LOWER,n,da,n,dw,&full_work));
    SOLVER(hipsolverDnZheevdx_bufferSize(h,HIPSOLVER_EIG_MODE_VECTOR,HIPSOLVER_EIG_RANGE_I,
        HIPBLAS_FILL_MODE_LOWER,n,da,n,0,0,1,nev,&found,dw,&partial_work));
    HIP(hipMalloc(&work,(size_t)std::max(full_work,partial_work)*sizeof(*work)));
    for(int mode=0;mode<4;mode++){
        bool active=mode==0 ? rank==0 : mode==1 ? rank<4 : mode==2 ? rank%block==0 : true;
        for(int driver=0;driver<2;driver++){
            MPI_Barrier(MPI_COMM_WORLD);
            double total=0,error=0,delta=0;int local_valid=1,info=0;
            if(active){
                for(int rep=0;rep<6;rep++){
                    HIP(hipMemcpyAsync(da,a.data(),a.size()*sizeof(a[0]),hipMemcpyHostToDevice,stream));
                    HIP(hipStreamSynchronize(stream));
                    auto start=std::chrono::steady_clock::now();
                    if(driver==0)SOLVER(hipsolverDnZheevd(h,HIPSOLVER_EIG_MODE_VECTOR,HIPBLAS_FILL_MODE_LOWER,
                        n,da,n,dw,work,full_work,di));
                    else SOLVER(hipsolverDnZheevdx(h,HIPSOLVER_EIG_MODE_VECTOR,HIPSOLVER_EIG_RANGE_I,
                        HIPBLAS_FILL_MODE_LOWER,n,da,n,0,0,1,nev,&found,dw,work,partial_work,di));
                    HIP(hipMemcpyAsync(&info,di,sizeof(int),hipMemcpyDeviceToHost,stream));
                    HIP(hipStreamSynchronize(stream));
                    if(rep)total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/5;
                    if(info || (driver && found!=nev))local_valid=0;
                }
                HIP(hipMemcpyAsync(w.data(),dw,n*sizeof(double),hipMemcpyDeviceToHost,stream));
                HIP(hipMemcpyAsync(v.data(),da,v.size()*sizeof(v[0]),hipMemcpyDeviceToHost,stream));
                HIP(hipStreamSynchronize(stream));
                error=residual(a,v,w,n,nev);
                if(driver==0)reference=w;
                else for(int i=0;i<nev;i++)delta=std::max(delta,std::abs(w[i]-reference[i]));
                if(!std::isfinite(error) || error>1e-10 || !std::isfinite(delta) || delta>1e-8)local_valid=0;
            }
            double worst_time,worst_error,worst_delta;int valid,count,one=active;
            MPI_Reduce(&total,&worst_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
            MPI_Reduce(&error,&worst_error,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
            MPI_Reduce(&delta,&worst_delta,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
            MPI_Reduce(&local_valid,&valid,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
            MPI_Reduce(&one,&count,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
            if(rank==0){printf("SOLVER_PROBE mode=%d active=%d n=%d nev=%d driver=%s seconds=%.6f residual=%.3e delta_eigenvalue=%.3e valid=%d workspace_mib=%.3f\n",
                mode,count,n,nev,driver?"heevdx":"heevd",worst_time,worst_error,worst_delta,valid,
                (driver?partial_work:full_work)*sizeof(hipDoubleComplex)/1048576.0);fflush(stdout);}
        }
    }
    HIP(hipFree(work));HIP(hipFree(di));HIP(hipFree(dw));HIP(hipFree(da));
    SOLVER(hipsolverDnDestroy(h));HIP(hipStreamDestroy(stream));MPI_Finalize();
}
