#pragma once
#include "openmx_core_reciprocal.h"
#include <memory>
#include <stdexcept>

namespace openmx::test::reciprocal {
void require(bool condition,const char *message) {
  if (!condition) throw std::runtime_error(message);
}
using Owner=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Owner context() {
  char error[128];
  Owner result(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  require(bool(result),error);return result;
}
struct Fixture {
  OpenMXPaoShape pao_shape{3,3,2,5,7};
  OpenMXVpsShape vps_shape{3,2,4,5,7};
  OpenMXNeutralProjectorShape neutral_shape{3,2,9};
  int lmax[3]{2,1,2}, n0[3]{2,0,1}, n1[2]{1,2}, n2[3]{};
  int *counts[3]{n0,n1,n2};
  OpenMXProductShape product_shape{3,9,lmax,counts};
  OpenMXPaoViews pao{};OpenMXVpsViews vps{};OpenMXProductViews products{};
  char error[256]{};
  int allocate(OpenMXCoreContext *core,int kind) {
    if (kind==0) return openmx_core_pao_allocate(core,&pao_shape,&pao,error,sizeof(error));
    if (kind==1) return openmx_core_vps_allocate(core,&vps_shape,&neutral_shape,&vps,error,sizeof(error));
    return openmx_core_product_allocate(core,&product_shape,&products,error,sizeof(error));
  }
  void allocate_all(OpenMXCoreContext *core) {
    for (int kind=0;kind<3;++kind) require(allocate(core,kind),error);
  }
};
OpenMXReciprocalViews get(OpenMXCoreContext *core) {
  char error[128];OpenMXReciprocalViews result{};
  require(openmx_core_reciprocal_get(core,&result,error,sizeof(error)),error);
  require(!error[0],"successful getter leaves error text");return result;
}
}
