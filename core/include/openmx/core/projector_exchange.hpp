#pragma once

#include <openmx/core/tensor.hpp>
#include <algorithm>
#include <climits>
#include <span>

namespace openmx::core {

// One synchronous projector exchange owns its cursors, message sizes and two
// independent buffers. Growing the receive buffer cannot invalidate an active
// send. The caller must complete a send before growing that direction again.
template<class Scalar>
class ProjectorExchange {
 public:
  explicit ProjectorExchange(int peers) : peers_(checked_peers(peers)), state_({4,peers_}) {}
  ProjectorExchange(const ProjectorExchange&) = delete;
  ProjectorExchange& operator=(const ProjectorExchange&) = delete;
  auto state() noexcept { return state_.view(); }
  auto state() const noexcept { return state_.view(); }
  std::span<Scalar> buffer(int direction, int elements) {
    if (direction < 0 || direction > 1 || elements < 0)
      throw std::invalid_argument("invalid projector exchange buffer request");
    auto& buffer = buffers_[direction];
    const auto size = std::max<std::size_t>(1,static_cast<std::size_t>(elements));
    if (buffer.size() < size) buffer.resize(size);
    return {buffer.data(),static_cast<std::size_t>(elements)};
  }
 private:
  static std::size_t checked_peers(int peers) {
    if (peers < 1) throw std::invalid_argument("projector exchange requires at least one peer");
    return static_cast<std::size_t>(peers);
  }
  std::size_t peers_;
  Tensor<int,2> state_;
  std::vector<Scalar> buffers_[2];
};

} // namespace openmx::core
