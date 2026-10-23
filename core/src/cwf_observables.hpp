#pragma once
#include <openmx_core_cwf.h>
#include <openmx/core/tensor.hpp>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace openmx::core {
class CwfObservables {
  struct Buffers {
    Tensor<double,2> charge,energy;
    Buffers(std::size_t spins,std::size_t orbitals)
        :charge({spins,orbitals}),energy({spins,orbitals}) {}
  };
 public:
  CwfObservables()=default;
  CwfObservables(const CwfObservables&)=delete;
  CwfObservables& operator=(const CwfObservables&)=delete;
  CwfObservables(CwfObservables&&)=delete;
  CwfObservables& operator=(CwfObservables&&)=delete;
  OpenMXCWFObservables* view() noexcept {return &view_;}
  void prepare(std::size_t spins,std::size_t orbitals) {
    if(spins!=1 && spins!=2)throw std::invalid_argument("CWF observables require one or two spin channels");
    if(!orbitals) {
      buffers_.reset();view_={spins,0,nullptr,nullptr,0};return;
    }
    if(buffers_ && view_.spin_channels==spins && view_.row_capacity==orbitals) {
      std::fill_n(buffers_->charge.data(),buffers_->charge.size(),0.0);
      std::fill_n(buffers_->energy.data(),buffers_->energy.size(),0.0);
      view_.distance=0;return;
    }
    auto replacement=std::make_unique<Buffers>(spins,orbitals);
    const OpenMXCWFObservables next={spins,orbitals,replacement->charge.data(),replacement->energy.data(),0};
    buffers_.swap(replacement);view_=next;
  }
  void clear() noexcept {buffers_.reset();view_={};}
 private:
  OpenMXCWFObservables view_{};
  std::unique_ptr<Buffers> buffers_;
};
}
