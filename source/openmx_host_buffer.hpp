#ifndef OPENMX_HOST_BUFFER_HPP
#define OPENMX_HOST_BUFFER_HPP
#include <cstdlib>
#include <memory>

namespace openmx::host {
/* Own malloc/calloc storage without a second allocation or initialization.
   C wire packets can transfer ownership with release(); borrowed views use
   get(). The deleter must match the allocator of every adopted pointer. */
struct Free {
    void operator()(void *pointer) const noexcept { std::free(pointer); }
};
template<class T> using Buffer=std::unique_ptr<T,Free>;
}
#endif
