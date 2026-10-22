#include "native_resources.hpp"
#include "calculation_context.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
template<class F> int checked(char* error,size_t capacity,F&& f) noexcept {
  if(error && capacity)*error='\0';
  try { f(); return 1; }
  catch(const std::exception& e) { if(error && capacity)std::snprintf(error,capacity,"%s",e.what()); }
  catch(...) { if(error && capacity)std::snprintf(error,capacity,"native resource ownership failed"); }
  return 0;
}
void require(bool ok,const char* message) { if(!ok)throw std::invalid_argument(message); }
constexpr int host_domain=0;
int free_host(void* pointer,int) { std::free(pointer);return 0; }
struct Release {
  OpenMXNativeResourceRelease callback=nullptr;
  int device=-1;
  OpenMXNativeResourceRelease synchronize=nullptr;
  void operator()(void* pointer) const noexcept { if(pointer && callback)(void)callback(pointer,device); }
};
using Resource=std::unique_ptr<void,Release>;
using Key=std::pair<int,std::uintptr_t>;
Key key(int domain,void* value) { return {domain,reinterpret_cast<std::uintptr_t>(value)}; }
struct State {
  size_t bytes;
  Resource storage;
  explicit State(size_t n):bytes(n),storage(nullptr,Release{free_host,-1}) {
    // malloc-family storage provides fundamental alignment and can acquire
    // the effective type of the native POD at the C ABI boundary.
    if(n>size_t(std::numeric_limits<std::ptrdiff_t>::max()))throw std::length_error("native state is too large");
    storage.reset(std::calloc(1,n?n:1));if(!storage)throw std::bad_alloc();
  }
};
}

struct OpenMXNativeResources {
  OpenMXNativeResourceReset reset;
  std::mutex mutex;
  std::map<Key,Resource> resources;
  std::map<std::string,State,std::less<>> states;
  explicit OpenMXNativeResources(OpenMXNativeResourceReset callback):reset(callback) {}
  ~OpenMXNativeResources() {
    // Native reset calls the same release operations; do not hold the mutex.
    if(reset)reset();
    // Fallback covers partial initialization and omitted native cleanup paths.
    for(auto& item:resources) {
      const auto& release=item.second.get_deleter();
      if(release.synchronize)(void)release.synchronize(item.second.get(),release.device);
    }
    while(!resources.empty())resources.erase(resources.begin());
  }
};

void NativeResourceGroups::Delete::operator()(OpenMXNativeResources* p) const noexcept { delete p; }

OpenMXNativeResources* NativeResourceGroups::get(const char* name,OpenMXNativeResourceReset reset) {
  require(name && *name,"missing native resource group name");
  auto found=groups.find(name);
  if(found!=groups.end()) {
    require(found->second->reset==reset,"native resource group has a different reset callback");
    return found->second.get();
  }
  Group owner(new OpenMXNativeResources(nullptr));auto* view=owner.get();
  // Publish the callback only after insertion succeeds.
  groups.emplace(name,std::move(owner));view->reset=reset;return view;
}

extern "C" int openmx_core_native_resources(OpenMXCoreContext* context,const char* name,
    OpenMXNativeResourceReset reset,OpenMXNativeResources** out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(context && name && *name && out,"missing native resource calculation/name/output");
    *out=context->native_resources.get(name,reset);
  });
}
extern "C" void openmx_core_native_resources_clear(OpenMXCoreContext* context) {
  if(context)context->native_resources.groups.clear();
}
extern "C" int openmx_core_native_adopt(OpenMXNativeResources* owner,int domain,void* value,
    int device,OpenMXNativeResourceRelease release,OpenMXNativeResourceRelease synchronize,
    char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(owner && domain>host_domain && release,"invalid native resource adoption");
    if(!value)return;
    std::lock_guard lock(owner->mutex);auto k=key(domain,value);
    require(owner->resources.find(k)==owner->resources.end(),"native resource already owned");
    // Allocate metadata before accepting ownership. Caller retains value if
    // insertion throws; no SDK call occurs on that failure path.
    auto [entry,inserted]=owner->resources.try_emplace(k,nullptr,Release{release,device,synchronize});
    require(inserted,"native resource insertion failed");entry->second.reset(value);
  });
}
extern "C" int openmx_core_native_release(OpenMXNativeResources* owner,int domain,void* value,int* status) {
  if(!owner || !value)return 0;
  Resource resource(nullptr,Release{});
  {
    std::lock_guard lock(owner->mutex);auto found=owner->resources.find(key(domain,value));
    if(found==owner->resources.end())return 0;
    resource=std::move(found->second);owner->resources.erase(found);
  }
  const auto release=resource.get_deleter();void* pointer=resource.release();
  const int result=release.callback(pointer,release.device);if(status)*status=result;return 1;
}
extern "C" void* openmx_core_native_malloc(OpenMXNativeResources* owner,size_t count,size_t bytes,int zero) {
  if(bytes && count>std::numeric_limits<size_t>::max()/bytes)return nullptr;
  if(count*bytes>size_t(std::numeric_limits<std::ptrdiff_t>::max()))return nullptr;
  if(!owner)return zero?std::calloc(count,bytes):std::malloc(count*bytes);
  void* pointer=zero?std::calloc(count,bytes):std::malloc(count*bytes);
  if(!pointer)return nullptr;
  Resource resource(pointer,Release{free_host,-1});
  try { std::lock_guard lock(owner->mutex);owner->resources.emplace(key(host_domain,pointer),std::move(resource));return pointer; }
  catch(...) { return nullptr; }
}
extern "C" void* openmx_core_native_realloc(OpenMXNativeResources* owner,void* pointer,size_t bytes) {
  if(bytes>size_t(std::numeric_limits<std::ptrdiff_t>::max()))return nullptr;
  if(!owner)return std::realloc(pointer,bytes);
  if(!pointer)return openmx_core_native_malloc(owner,1,bytes,0);
  std::lock_guard lock(owner->mutex);auto found=owner->resources.find(key(host_domain,pointer));
  if(found==owner->resources.end())return nullptr;
  if(!bytes) { owner->resources.erase(found);return nullptr; }
  void* replacement=std::realloc(pointer,bytes);if(!replacement)return nullptr;
  // Reuse the existing map node, so moving realloc cannot fail to register.
  auto node=owner->resources.extract(found);(void)node.mapped().release();
  node.key()=key(host_domain,replacement);node.mapped().reset(replacement);
  owner->resources.insert(std::move(node));return replacement;
}
extern "C" void openmx_core_native_free(OpenMXNativeResources* owner,void* pointer) {
  int status;if(!openmx_core_native_release(owner,host_domain,pointer,&status))std::free(pointer);
}
extern "C" int openmx_core_native_state(OpenMXNativeResources* owner,const char* name,size_t bytes,
    void** out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(owner && name && *name && out,"missing native state owner/name/output");
    std::lock_guard lock(owner->mutex);auto found=owner->states.find(name);
    if(found!=owner->states.end())require(found->second.bytes==bytes,"native state size changed");
    else found=owner->states.try_emplace(name,bytes).first;
    *out=found->second.storage.get();
  });
}
extern "C" size_t openmx_core_native_resource_count(OpenMXNativeResources* owner) {
  if(!owner)return 0;std::lock_guard lock(owner->mutex);return owner->resources.size();
}
