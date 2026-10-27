#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include "mpi.h"
#include "openmx_core_solver_workspace.h"
#include "openmx_core_projector_exchange.h"
#define CHECK(c) do{if(!(c)){std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);std::exit(2);}}while(0)
static int destroyed,expected_abort_destroys;
static bool inject_adoption_failure;
static const void *abort_output;
static unsigned char abort_snapshot[256];
/* Forwarding instrumentation: production objects and resource groups are real. */
static void tracked_destroy(OpenMXProjectorExchange *owner){CHECK(owner);++destroyed;openmx_core_projector_exchange_destroy(owner);}
static int tracked_adopt(OpenMXNativeResources *group,int domain,void *value,int device,
    OpenMXNativeResourceRelease release,OpenMXNativeResourceRelease synchronize,char *error,size_t capacity){
  if(inject_adoption_failure){inject_adoption_failure=false;std::snprintf(error,capacity,"injected adoption failure");return 0;}
  return openmx_core_native_adopt(group,domain,value,device,release,synchronize,error,capacity);
}
static int discard_adoption(OpenMXNativeResources*,int,void*,int,
    OpenMXNativeResourceRelease,OpenMXNativeResourceRelease,char*,size_t){return 1;}
#define openmx_core_projector_exchange_destroy tracked_destroy
#define openmx_core_native_adopt tracked_adopt
#include "openmx_projector_exchange.h"
#undef openmx_core_projector_exchange_destroy
#undef openmx_core_native_adopt
int MPI_Abort(MPI_Comm communicator,int code){
  CHECK(communicator==3&&code==EXIT_FAILURE&&destroyed==expected_abort_destroys);
  CHECK(abort_output&&std::memcmp(abort_output,abort_snapshot,sizeof(OpenMXProjectorTransfer))==0);
  std::_Exit(77);
}
struct Arguments {int peers,exit_mode;const char *invalid;};
static OpenMXNativeResources *group(OpenMXSolverInvocation *owner,const char *name){
  OpenMXNativeResources *result=nullptr;char error[256];
  CHECK(openmx_core_solver_resources(owner,name,&result,error,sizeof(error))&&result);return result;
}
template<class Scalar>static void check_transfer(OpenMXProjectorTransfer &transfer,int peers){
  CHECK(transfer.owner&&transfer.communicator==3);
  int *cursors[4]={transfer.buffers.sent,transfer.buffers.received,transfer.buffers.send_sizes,transfer.buffers.receive_sizes};
  for(int c=0;c<4;c++){
    CHECK(cursors[c]);for(int p=0;p<peers;p++)CHECK(cursors[c][p]==0);
    for(int other=0;other<c;other++)CHECK(cursors[c]!=cursors[other]);
  }
  auto *empty=static_cast<Scalar*>(OpenMX_TransferBuffer(&transfer,OPENMX_EXCHANGE_SEND,0));CHECK(empty);
  auto *send=static_cast<Scalar*>(OpenMX_TransferBuffer(&transfer,OPENMX_EXCHANGE_SEND,7));
  for(int i=0;i<7;i++)send[i]=static_cast<Scalar>(.125*(i+1));
  auto *receive=static_cast<Scalar*>(OpenMX_TransferBuffer(&transfer,OPENMX_EXCHANGE_RECEIVE,8193));
  CHECK(receive&&receive!=send);receive[8192]=Scalar(19);
  CHECK(send==OpenMX_TransferBuffer(&transfer,OPENMX_EXCHANGE_SEND,3));
  for(int i=0;i<7;i++)CHECK(send[i]==static_cast<Scalar>(.125*(i+1)));
  CHECK(OpenMX_TransferCount(&transfer,4,3,7)==84&&OpenMX_TransferCount(&transfer,0,3,7)==0);
}
static int operation(OpenMXSolverInvocation *owner,void *opaque,double *result){
  auto &args=*static_cast<Arguments*>(opaque);auto *first_group=group(owner,"test:float-exchange");
  CHECK(openmx_core_native_resource_count(first_group)==0);
  if(args.invalid){
    OpenMXProjectorTransfer output;std::memset(&output,0xa5,sizeof(output));abort_output=&output;
    static_assert(sizeof(output)<=sizeof(abort_snapshot));std::memcpy(abort_snapshot,&output,sizeof(output));
    expected_abort_destroys=!std::strcmp(args.invalid,"adopt")?1:0;inject_adoption_failure=expected_abort_destroys!=0;
    OpenMX_TransferBeginOwned(!std::strcmp(args.invalid,"resources")?nullptr:first_group,3,args.peers,
        !std::strcmp(args.invalid,"create")?1:sizeof(float),!std::strcmp(args.invalid,"output")?nullptr:&output);CHECK(false);
  }
  OpenMXProjectorTransfer first,second;
  OpenMX_TransferBeginOwned(first_group,3,args.peers,sizeof(float),&first);
  auto *second_group=group(owner,"test:double-exchange");
  OpenMX_TransferBeginOwned(second_group,3,args.peers,sizeof(double),&second);
  CHECK(first.owner!=second.owner&&first_group!=second_group);
  CHECK(openmx_core_native_resource_count(first_group)==1&&openmx_core_native_resource_count(second_group)==1);
  CHECK(first_group==group(owner,"test:float-exchange"));
  check_transfer<float>(first,args.peers);check_transfer<double>(second,args.peers);
  CHECK(destroyed==0);
  if(args.exit_mode==1)return 0;
  if(args.exit_mode==2)throw std::runtime_error("injected callback unwind");
  *result=17;return 1;
}
int main(int argc,char **argv){
  char error[256];Arguments args={1,0,argc>1?argv[1]:nullptr};double result=59;
  if(args.invalid){destroyed=0;(void)openmx_core_solver_run(1,operation,&args,&result,error,sizeof(error));CHECK(false);}
  int cases=0;
  for(int peers:{1,4,1})for(int exit_mode=0;exit_mode<3;exit_mode++){
    args={peers,exit_mode,nullptr};destroyed=0;result=59;
    const int success=openmx_core_solver_run(1,operation,&args,&result,error,sizeof(error));
    CHECK(destroyed==2&&success==(exit_mode==0));
    if(!exit_mode)CHECK(result==17&&error[0]=='\0');
    else{CHECK(result==59&&error[0]);if(exit_mode==2)CHECK(std::strstr(error,"injected callback unwind"));}
    ++cases;
  }
  destroyed=0;auto legacy=OpenMX_TransferBegin(3,2,sizeof(double));check_transfer<double>(legacy,2);
  CHECK(destroyed==0);OpenMX_TransferEnd(&legacy);CHECK(destroyed==1&&!legacy.owner);
  std::printf("PASS: %d scoped float/double exchange cases, normal/failed/throwing exits, exact destruction, buffer stability and legacy caller compatibility\n",cases);
}
