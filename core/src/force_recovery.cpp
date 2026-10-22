#include <openmx/core/force_recovery.hpp>
#include <cstdio>
#include <stdexcept>

namespace {
std::size_t items_extent(std::size_t items) {
  if(items>OPENMX_FORCE_RECOVERY_MAX_ITEMS)
    throw std::length_error("Force recovery batch exceeds the item limit");
  return items+1;
}
std::size_t nonempty(std::size_t count) noexcept { return count?count:1; }
template<class F> int checked(char* error,std::size_t capacity,F&& operation) noexcept {
  if(error && capacity)*error='\0';
  try { operation();return 1; }
  catch(const std::exception& e) { if(error && capacity)std::snprintf(error,capacity,"%s",e.what()); }
  catch(...) { if(error && capacity)std::snprintf(error,capacity,"unexpected Force recovery failure"); }
  return 0;
}
}
namespace openmx::core {
ForceVnaRecoveryPacket::ForceVnaRecoveryPacket(std::size_t items,std::size_t values,std::size_t density_values)
    :items_(items_extent(items)-1),values_(values),density_values_(density_values),
     offset_({items_+1}),density_offset_({items_+1}),
     projectors_({nonempty(values)}),density_({nonempty(density_values)}) {}
OpenMXForceVnaRecoveryView ForceVnaRecoveryPacket::view() noexcept {
  return {items_,values_,density_values_,offset_.data(),density_offset_.data(),projectors_.data(),density_.data()};
}
ForceHnlRecoveryPacket::ForceHnlRecoveryPacket(std::size_t items,std::size_t values,std::size_t lengths)
    :items_(items_extent(items)-1),values_(values),lengths_(lengths),
     item_({nonempty(items_)}),offset_({items_+1}),length_offset_({items_+1}),
     data_({nonempty(values)}),length_({nonempty(lengths)}) {}
OpenMXForceHnlRecoveryView ForceHnlRecoveryPacket::view() noexcept {
  return {items_,values_,lengths_,item_.data(),offset_.data(),length_offset_.data(),data_.data(),length_.data()};
}
}
extern "C" int openmx_core_force_vna_recovery_run(size_t items,size_t values,size_t density_values,
    OpenMXForceVnaRecoveryOperation operation,void* arguments,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    if(!operation)throw std::invalid_argument("missing VNA Force recovery callback");
    openmx::core::ForceVnaRecoveryPacket packet(items,values,density_values);const auto view=packet.view();
    if(operation(&view,arguments)!=1)throw std::runtime_error("VNA Force recovery callback failed");
  });
}
extern "C" int openmx_core_force_hnl_recovery_run(size_t items,size_t values,size_t lengths,
    OpenMXForceHnlRecoveryOperation operation,void* arguments,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    if(!operation)throw std::invalid_argument("missing nonlocal Force recovery callback");
    openmx::core::ForceHnlRecoveryPacket packet(items,values,lengths);const auto view=packet.view();
    if(operation(&view,arguments)!=1)throw std::runtime_error("nonlocal Force recovery callback failed");
  });
}
