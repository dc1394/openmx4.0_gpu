#include "openmx_core_grid.h"
#include "host_simd_runtime.hpp"
#include <iostream>

extern "C" int openmx_test_abi_sve128();
extern "C" int openmx_test_abi_sve256();
extern "C" int openmx_test_abi_sve512();

int main() {
  int executed=0,result=0;
  auto run=[&](int mode,int bits,int (*test)()) {
    if(!openmx::core::host::openmx_host_simd_usable(mode)) {
      std::cout<<"SVE"<<bits<<": unavailable at the current thread VL; not executed\n";
      return;
    }
    ++executed;
    const int status=test();result|=status;
    std::cout<<"SVE"<<bits<<": "<<(status?"FAIL":"PASS")<<'\n';
  };
#ifdef OPENMX_HOST_SIMD_SVE128
  run(OPENMX_GRID_SVE128,128,openmx_test_abi_sve128);
#endif
#ifdef OPENMX_HOST_SIMD_SVE256
  run(OPENMX_GRID_SVE256,256,openmx_test_abi_sve256);
#endif
#ifdef OPENMX_HOST_SIMD_SVE512
  run(OPENMX_GRID_SVE512,512,openmx_test_abi_sve512);
#endif
  if(executed>1)return 1; // fixed-size variants require an exact VL match
  return executed?result:77;
}
