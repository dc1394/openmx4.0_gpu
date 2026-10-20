#include <openmx/core/projector_exchange.hpp>
#include "openmx_core_projector_exchange.h"
#include <cstdio>
#include <memory>
#include <variant>

struct OpenMXProjectorExchange {
  using Float = openmx::core::ProjectorExchange<float>;
  using Double = openmx::core::ProjectorExchange<double>;
  OpenMXProjectorExchange(int peers, int scalar) : storage(make(peers,scalar)) {}
  static std::variant<Float,Double> make(int peers,int scalar) {
    if (scalar==OPENMX_EXCHANGE_FLOAT) return std::variant<Float,Double>(std::in_place_type<Float>,peers);
    if (scalar==OPENMX_EXCHANGE_DOUBLE) return std::variant<Float,Double>(std::in_place_type<Double>,peers);
    throw std::invalid_argument("unsupported projector exchange scalar");
  }
  std::variant<Float,Double> storage;
};

namespace {
template<class Function>
int checked(char *error,std::size_t capacity,Function&& function) noexcept {
  if (error && capacity) error[0]='\0';
  try { function(); return 1; }
  catch (const std::exception& e) { if (error && capacity) std::snprintf(error,capacity,"%s",e.what()); }
  catch (...) { if (error && capacity) std::snprintf(error,capacity,"unknown projector exchange failure"); }
  return 0;
}
}

extern "C" OpenMXProjectorExchange *openmx_core_projector_exchange_create(int peers,int scalar,
    OpenMXProjectorExchangeBuffers *buffers,char *error,std::size_t capacity)
{
  std::unique_ptr<OpenMXProjectorExchange> result;
  checked(error,capacity,[&] {
    if (!buffers) throw std::invalid_argument("null projector exchange views");
    auto next=std::make_unique<OpenMXProjectorExchange>(peers,scalar);
    std::visit([&](auto& owner) {
      auto state=owner.state();auto* data=state.data_handle();
      *buffers={data,data+peers,data+2*static_cast<std::size_t>(peers),data+3*static_cast<std::size_t>(peers)};
    },next->storage);
    result=std::move(next);
  });
  return result.release();
}

extern "C" int openmx_core_projector_exchange_buffer(OpenMXProjectorExchange *exchange,
    int direction,int elements,void **buffer,char *error,std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!exchange || !buffer) throw std::invalid_argument("null projector exchange request");
    std::visit([&](auto& owner) { *buffer=owner.buffer(direction,elements).data(); },exchange->storage);
  });
}

extern "C" void openmx_core_projector_exchange_destroy(OpenMXProjectorExchange *exchange) { delete exchange; }
