#include <openmx/core/pivoted_inverse.hpp>
#include "openmx_core_solver_workspace.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {
void check(bool valid,const char* message){if(!valid)throw std::runtime_error(message);}
struct Oracle {
  int factors=0,inverses=0,phase=0,n=0,tag=713,factor_info=0,invert_info=0,throw_at=0;
  double* packed=nullptr;int* pivots=nullptr;
  std::array<double,64> expected{},original{};
  OpenMXRealLuOperations operations(){return {this,factor,invert};}
  void input(const double* values,int count,int stride){
    phase=0;for(int i=0;i<count;++i)for(int j=0;j<count;++j)expected[i*count+j]=values[i*stride+j];
  }
  static void factor(void* opaque,int* m,int* n,double* a,int* lda,int* pivots,int* info){
    auto& o=*static_cast<Oracle*>(opaque);
    check(o.phase==0 && *m==*n && *n>=0 && *n<=8 && *lda==*n,"factor dimensions or order changed");
    o.phase=1;o.n=*n;o.packed=a;o.pivots=pivots;o.factors++;
    for(int i=0;i<*n;++i)for(int j=0;j<*n;++j)check(a[i*(*n)+j]==o.expected[i*(*n)+j],"native row packing was transposed");
    for(int i=0;i<*n;++i){pivots[i]=*n-i;
      for(int j=0;j<*n;++j){o.original[i*8+j]=a[j*(*n)+i];a[j*(*n)+i]=71+i*8+j;}}
    // Deliberately probe the documented legacy extra row/column and pivot.
    a[(*n+1)*(*n+1)-1]=o.tag;pivots[*n]=o.tag;*info=o.factor_info;
    if(o.throw_at==1)throw std::runtime_error("injected factor callback exception");
  }
  static void invert(void* opaque,int* n,double* a,int* lda,int* pivots,double* work,int* lwork,int* info){
    auto& o=*static_cast<Oracle*>(opaque);
    check(o.phase==1 && *n==o.n && *lda==*n && *lwork==*n,"inverse dimensions or order changed");
    check(a==o.packed && pivots==o.pivots && pivots[*n]==o.tag && a[(*n+1)*(*n+1)-1]==o.tag,"factor buffers/padding did not survive");
    o.inverses++;work[*n]=o.tag;*info=o.invert_info;
    if(o.throw_at==2)throw std::runtime_error("injected inverse callback exception");
    long double augmented[8][16]={};
    for(int i=0;i<*n;++i){check(pivots[i]==*n-i,"pivot storage changed");work[i]=i+.25;
      for(int j=0;j<*n;++j){check(a[j*(*n)+i]==71+i*8+j,"factor matrix changed before inverse");
        augmented[i][j]=o.original[i*8+j];augmented[i][*n+j]=i==j;}}
    // Independent long-double Gauss-Jordan oracle for the LAPACK result.
    for(int k=0;k<*n;++k){
      int pivot=k;for(int i=k+1;i<*n;++i)if(std::abs(augmented[i][k])>std::abs(augmented[pivot][k]))pivot=i;
      check(std::abs(augmented[pivot][k])>1e-15L,"singular oracle matrix");
      for(int j=0;j<2*(*n);++j)std::swap(augmented[k][j],augmented[pivot][j]);
      const auto scale=augmented[k][k];for(int j=0;j<2*(*n);++j)augmented[k][j]/=scale;
      for(int i=0;i<*n;++i)if(i!=k){const auto factor=augmented[i][k];
        for(int j=0;j<2*(*n);++j)augmented[i][j]-=factor*augmented[k][j];}
    }
    for(int i=0;i<*n;++i)for(int j=0;j<*n;++j)a[j*(*n)+i]=double(augmented[i][*n+j]);
    o.phase=0;
  }
};
struct Matrices {
  std::array<double,80> a,b;
  OpenMXSolverRealTensor input={{8,10,1,1},a.data()},output={{10,8,1,1},b.data()};
  void seed(int n){a.fill(-71);b.fill(-91);for(int i=0;i<n;++i)for(int j=0;j<n;++j)a[i*10+j]=i==j?4+i:.125*(1+i+2*j);}
  void identity(int n,const std::array<double,80>& before){
    check(a==before,"pivoted inverse changed input");
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){long double sum=0;
      for(int k=0;k<n;++k)sum+=static_cast<long double>(before[i*10+k])*b[k*8+j];
      check(std::isfinite(sum) && std::abs(sum-(i==j?1.L:0.L))<1e-11L,"independent inverse identity differs");}
    for(int i=0;i<10;++i)for(int j=0;j<8;++j)if(i>=n || j>=n)check(b[i*8+j]==-91,"inactive output changed");
  }
};
void apply(OpenMXSolverInvocation* owner,int worker,Matrices& matrix,Oracle& oracle,int n){
  oracle.input(matrix.a.data(),n,10);auto ops=oracle.operations();char error[256];
  check(openmx_core_solver_inverse_pivoted(owner,worker,n-1,&matrix.input,&matrix.output,&ops,error,sizeof(error)),error);
}
void views(OpenMXSolverInvocation* owner){
  Matrices m;Oracle o;
  for(int n:{1,3,2,4,3,6}){m.seed(n);const auto saved=m.a;apply(owner,0,m,o,n);m.identity(n,saved);}
  m.seed(3);const auto saved=m.a;o.input(m.a.data(),3,10);auto ops=o.operations();
  openmx::core::PivotedInverse direct(3);
  direct.apply(openmx::core::PivotedInverse::matrix_view{m.a.data(),8,10},
      openmx::core::PivotedInverse::matrix_view{m.b.data(),10,8},3,ops);m.identity(3,saved);
}
void reuse(OpenMXSolverInvocation* owner){
  Matrices m;Oracle o;m.seed(4);apply(owner,0,m,o,4);auto* held=o.packed;
  for(int n:{3,1,4,2}){m.seed(n);const auto saved=m.a;o.input(m.a.data(),n,10);auto ops=o.operations();char error[256];
    openmx::test::fail_allocation_after(0);
    const int ok=openmx_core_solver_inverse_pivoted(owner,0,n-1,&m.input,&m.output,&ops,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    check(ok && o.packed==held,"pivoted capacity reuse allocated or changed storage");m.identity(n,saved);
  }
}
void failures(OpenMXSolverInvocation* owner){
  Matrices m;Oracle o;m.seed(4);apply(owner,0,m,o,4);auto* held=o.packed;
  for(long fail=0;fail<4;++fail){
    m.seed(6);const auto saved=m.a,output=m.b;const auto live=openmx::test::live_allocations();
    o.input(m.a.data(),6,10);auto ops=o.operations();char error[256];const int calls=o.factors;
    openmx::test::fail_allocation_after(fail);
    const int ok=openmx_core_solver_inverse_pivoted(owner,0,5,&m.input,&m.output,&ops,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    check(!ok && error[0] && o.factors==calls && m.a==saved && m.b==output && openmx::test::live_allocations()==live,"failed pivoted growth changed arrays/called LAPACK/leaked");
    m.seed(3);o.input(m.a.data(),3,10);const auto original=m.a;openmx::test::fail_allocation_after(0);
    const int reused=openmx_core_solver_inverse_pivoted(owner,0,2,&m.input,&m.output,&ops,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);check(reused && o.packed==held,"failed growth discarded old pivoted capacity");m.identity(3,original);
  }
  m.seed(6);const auto original=m.a;apply(owner,0,m,o,6);m.identity(6,original);held=o.packed;
  m.seed(2);apply(owner,1,m,o,2);check(o.packed!=held,"workers share pivoted scratch");
  for(int bad=0;bad<19;++bad){
    m.seed(2);auto input=m.input,output=m.output;auto ops=o.operations();int n=1,worker=0;auto* scope=owner;auto* operation=&ops;auto* input_view=&input;auto* output_view=&output;
    if(bad==0)n=-2;if(bad==1)n=8;if(bad==2)input.extent[2]=2;if(bad==3)output.extent[3]=2;
    if(bad==4)input.extent[0]=SIZE_MAX;if(bad==5)input.values=nullptr;if(bad==6)output.values=nullptr;
    if(bad==7)output.extent[1]=1;if(bad==8)output.values=m.a.data();
    if(bad==9){output.values=m.a.data()+3;output.extent[0]=7;}
    if(bad==10)ops.factor=nullptr;if(bad==11)ops.invert=nullptr;if(bad==12)operation=nullptr;
    if(bad==13)worker=-1;if(bad==14)scope=nullptr;
    if(bad==15)input_view=nullptr;if(bad==16)output_view=nullptr;if(bad==17)worker=4;if(bad==18)n=INT_MAX;
    const auto a=m.a,b=m.b;const int calls=o.factors;char error[256];
    check(!openmx_core_solver_inverse_pivoted(scope,worker,n,input_view,output_view,operation,error,sizeof(error)) && error[0],"invalid pivoted inverse accepted");
    check(m.a==a && m.b==b && o.factors==calls,"invalid pivoted request changed arrays or called LAPACK");
  }
}
void callbacks(OpenMXSolverInvocation* owner){
  Matrices m;Oracle first,second;m.seed(2);apply(owner,0,m,first,2);const int calls=first.factors;
  second.tag=1731;m.seed(2);const auto original=m.a;apply(owner,0,m,second,2);m.identity(2,original);
  check(first.factors==calls && second.factors==1 && second.inverses==1,"pivoted owner cached the old provider context");
  for(int where:{1,2}){m.seed(3);const auto a=m.a,b=m.b;second.throw_at=where;second.input(m.a.data(),3,10);auto ops=second.operations();char error[256];
    check(!openmx_core_solver_inverse_pivoted(owner,0,2,&m.input,&m.output,&ops,error,sizeof(error)) && std::strstr(error,"injected"),"callback exception escaped its C ABI");
    check(m.a==a && m.b==b,"callback exception changed public arrays");
  }
  second.throw_at=0;second.factor_info=1;second.invert_info=2;m.seed(2);const auto a=m.a;
  const int before=second.inverses;apply(owner,0,m,second,2);m.identity(2,a);
  check(second.inverses==before+1,"nonzero INFO skipped the legacy inversion call");
}
void empty(OpenMXSolverInvocation* owner){
  Matrices m;m.seed(0);const auto a=m.a,b=m.b;Oracle o;o.factor_info=o.invert_info=-4;
  OpenMXSolverRealTensor zero={{0,0,1,1},nullptr};auto ops=o.operations();char error[256];
  check(openmx_core_solver_inverse_pivoted(owner,2,-1,&zero,&m.output,&ops,error,sizeof(error)),error);
  check(o.factors==1 && o.inverses==1 && o.n==0 && m.a==a && m.b==b,"empty legacy calls changed arrays/order/dimensions");
  openmx::test::fail_allocation_after(0);
  const int reused=openmx_core_solver_inverse_pivoted(owner,2,-1,&zero,&zero,&ops,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);check(reused && o.factors==2 && o.inverses==2,"empty pivoted reuse allocated");
}
void concurrent(OpenMXSolverInvocation* owner){
  std::vector<std::thread> threads;
  for(int worker=0;worker<4;++worker)threads.emplace_back([=]{Matrices m;Oracle o;o.tag+=worker;
    for(int repeat=0;repeat<30;++repeat){int n=(worker+repeat)%4+1;m.seed(n);const auto a=m.a;apply(owner,worker,m,o,n);m.identity(n,a);}
    check(o.factors==30 && o.inverses==30,"concurrent provider context was lost");
  });
  for(auto& thread:threads)thread.join();
}
int lifetime_callback(OpenMXSolverInvocation* owner,void* opaque,double* result){
  Matrices m;Oracle o;m.seed(3);const auto original=m.a;apply(owner,0,m,o,3);m.identity(3,original);
  const int mode=*static_cast<int*>(opaque);
  if(mode==1)return 0;
  if(mode==2)throw std::runtime_error("injected outer callback exception");
  *result=17;return 1;
}
void lifetime(){
  char error[256];
  for(int mode=0;mode<3;++mode){double result=-19;const auto before=openmx::test::live_allocations();
    const int ok=openmx_core_solver_run(4,lifetime_callback,&mode,&result,error,sizeof(error));
    check(ok==(mode==0) && result==(mode==0?17:-19) && openmx::test::live_allocations()==before,"pivoted scope leaked on return/early return/exception");
  }
  int mode=0;double result;openmx::test::fail_allocation_after(10000);
  check(openmx_core_solver_run(4,lifetime_callback,&mode,&result,error,sizeof(error)),error);
  const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  check(count>=5,"pivoted allocation interception missed actual ownership");
  for(long fail=0;fail<count;++fail){result=-19;const auto before=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(fail);
    const int ok=openmx_core_solver_run(4,lifetime_callback,&mode,&result,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    check(!ok && error[0] && result==-19 && openmx::test::live_allocations()==before,"partial pivoted scope construction leaked or changed result");
  }
}
int operation(OpenMXSolverInvocation* owner,void* opaque,double* result){
  const auto name=*static_cast<std::string_view*>(opaque);
  if(name=="views")views(owner);else if(name=="reuse")reuse(owner);
  else if(name=="failures")failures(owner);else if(name=="callbacks")callbacks(owner);
  else if(name=="empty")empty(owner);else if(name=="concurrent")concurrent(owner);
  else throw std::invalid_argument("unknown pivoted inverse test");
  *result=17;return 1;
}
}
int main(int argc,char** argv){try{
  check(argc==2,"select a pivoted inverse test");std::string_view name=argv[1];
  const auto before=openmx::test::live_allocations();openmx::test::fill_allocations_with(0xff);
  if(name=="lifetime")lifetime();
  else {double result=-19;char error[256];check(openmx_core_solver_run(4,operation,&name,&result,error,sizeof(error)),error);check(result==17,"pivoted scope result changed");}
  openmx::test::fill_allocations_with(-1);
  check(openmx::test::live_allocations()==before,"pivoted inverse test leaked");
  std::cout<<"PASS: pivoted inverse "<<name<<'\n';return 0;
}catch(const std::exception& error){openmx::test::fail_allocation_after(-1);openmx::test::fill_allocations_with(-1);std::cerr<<error.what()<<'\n';return 1;}}
