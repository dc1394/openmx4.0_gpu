#pragma once
#include <openmx/core/tensor.hpp>
#include "openmx_core_force_recovery.h"

namespace openmx::core {
class ForceVnaRecoveryPacket {
 public:
  ForceVnaRecoveryPacket(std::size_t items, std::size_t values, std::size_t density_values);
  ForceVnaRecoveryPacket(const ForceVnaRecoveryPacket&)=delete;
  ForceVnaRecoveryPacket& operator=(const ForceVnaRecoveryPacket&)=delete;
  OpenMXForceVnaRecoveryView view() noexcept;
 private:
  std::size_t items_, values_, density_values_;
  Tensor<std::size_t,1> offset_, density_offset_;
  Tensor<float,1> projectors_;
  Tensor<double,1> density_;
};
class ForceHnlRecoveryPacket {
 public:
  ForceHnlRecoveryPacket(std::size_t items, std::size_t values, std::size_t lengths);
  ForceHnlRecoveryPacket(const ForceHnlRecoveryPacket&)=delete;
  ForceHnlRecoveryPacket& operator=(const ForceHnlRecoveryPacket&)=delete;
  OpenMXForceHnlRecoveryView view() noexcept;
 private:
  std::size_t items_, values_, lengths_;
  Tensor<OpenMXForceHnlRecoveryItem,1> item_;
  Tensor<std::size_t,1> offset_, length_offset_;
  Tensor<double,1> data_;
  Tensor<int,1> length_;
};
} // namespace openmx::core
