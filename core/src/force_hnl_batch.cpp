#include <openmx/core/force_hnl_batch.hpp>
#include <climits>
#include <cstdio>
#include <stdexcept>

namespace {
std::size_t nonempty(std::size_t count) noexcept { return count?count:1; }
int validate(int kind,std::size_t items,std::size_t rows) {
  if(kind!=1 && kind!=2)throw std::invalid_argument("invalid nonlocal Force batch kind");
  if(items>INT_MAX || rows>INT_MAX)throw std::length_error("nonlocal Force batch exceeds kernel index range");
  if(kind==2 && rows!=items)throw std::invalid_argument("nonlocal Force case 2 requires one row per pair");
  return kind;
}
class Finish {
 public:
  Finish(OpenMXForceHnlBatchFinish callback,void* arguments):callback_(callback),arguments_(arguments) {}
  ~Finish() { if(callback_)callback_(arguments_); }
  Finish(const Finish&)=delete;
  Finish& operator=(const Finish&)=delete;
 private:
  OpenMXForceHnlBatchFinish callback_;
  void* arguments_;
};
}
namespace openmx::core {
ForceHnlBatch::ForceHnlBatch(int kind,std::size_t items,std::size_t rows,std::size_t density_values)
    :kind_(validate(kind,items,rows)),items_(items),rows_(rows),density_values_(density_values),
     item_({nonempty(items)}),atoms_({kind==1?2U:3U,nonempty(items)}),
     tags_({3,nonempty(rows)}),offsets_({2,nonempty(rows)}),
     density_({nonempty(density_values)}),force_({nonempty(items),3}) {}
OpenMXForceHnlBatchView ForceHnlBatch::view() noexcept {
  const auto n=nonempty(items_),r=nonempty(rows_);
  return {kind_,items_,rows_,density_values_,item_.data(),atoms_.data(),atoms_.data()+n,
      kind_==2?atoms_.data()+2*n:nullptr,offsets_.data(),offsets_.data()+r,
      tags_.data(),tags_.data()+r,tags_.data()+2*r,density_.data(),force_.data()};
}
}
extern "C" int openmx_core_force_hnl_batch_run(int kind,size_t items,size_t rows,size_t density_values,
    OpenMXForceHnlBatchOperation operation,OpenMXForceHnlBatchFinish finish,void* arguments,char* error,size_t capacity) {
  if(error && capacity)*error='\0';
  try {
    if(!operation)throw std::invalid_argument("missing nonlocal Force batch callback");
    openmx::core::ForceHnlBatch batch(kind,items,rows,density_values);
    const auto view=batch.view();
    Finish cleanup(finish,arguments);
    if(operation(&view,arguments)!=1)throw std::runtime_error("nonlocal Force batch callback failed");
    return 1;
  }catch(const std::exception& e) {
    if(error && capacity)std::snprintf(error,capacity,"%s",e.what());
  }catch(...) {
    if(error && capacity)std::snprintf(error,capacity,"unexpected nonlocal Force batch failure");
  }
  return 0;
}
