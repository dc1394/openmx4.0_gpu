#pragma once
#include "openmx_core_olg_cache.h"
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace openmx::core {
// Native GListTAtoms consumers still free each row/table separately. Keep
// malloc-compatible arrays under RAII until that explicit ownership transfer;
// ordinary C++ containers own all cache-only metadata. The allocation service
// is private to this host owner and must return free()-compatible storage.
class OlgCache {
 public:
  using Allocate=void*(*)(std::size_t);
  explicit OlgCache(const OpenMXOlgCacheInput& input,Allocate allocate=std::malloc)
      :md_iter_(input.md_iter),copied_cells_(input.copied_cells),
       global_atoms_(input.global_atoms),allocate_(allocate) {
    validate(input);
    if(!allocate_)throw std::invalid_argument("missing overlap-list host allocator");
    atoms_.reserve(input.local_atoms);
    for(std::size_t a=0;a<input.local_atoms;++a) {
      const int global=input.local_to_global[a+1];
      atoms_.emplace_back(global,std::size_t(input.neighbor_counts[global])+1,allocate_);
    }
  }
  OlgCache(const OlgCache&)=delete;
  OlgCache& operator=(const OlgCache&)=delete;
  OlgCache(OlgCache&&)=delete;
  OlgCache& operator=(OlgCache&&)=delete;

  static void validate(const OpenMXOlgCacheInput& input) {
    if(input.md_iter<0 || input.copied_cells<0 || input.global_atoms>std::size_t(INT_MAX) ||
       input.local_atoms>input.global_atoms)
      throw std::invalid_argument("invalid overlap-list cache identity");
    if(input.local_atoms && (!input.local_to_global || !input.neighbor_counts))
      throw std::invalid_argument("missing overlap-list atom map/counts");
    for(std::size_t a=0;a<input.local_atoms;++a) {
      const int global=input.local_to_global[a+1];
      if(global<=0 || std::size_t(global)>input.global_atoms)
        throw std::invalid_argument("invalid overlap-list global atom index");
      if(input.neighbor_counts[global]<0 || input.neighbor_counts[global]==INT_MAX)
        throw std::invalid_argument("invalid overlap-list neighbour count");
    }
  }
  bool matches(const OpenMXOlgCacheInput& input) const {
    // The C ABI validates the borrowed input before querying even an absent
    // cache; callers of this internal method provide the same valid input.
    if(md_iter_!=input.md_iter || copied_cells_!=input.copied_cells ||
       global_atoms_!=input.global_atoms || atoms_.size()!=input.local_atoms)return false;
    for(std::size_t a=0;a<atoms_.size();++a) {
      const auto& atom=atoms_[a];
      const int global=input.local_to_global[a+1];
      if(atom.taken || atom.global!=global || atom.pairs.size()!=std::size_t(input.neighbor_counts[global])+1 ||
         !complete(atom))return false;
    }
    return true;
  }
  void store(std::size_t a,std::size_t p,std::size_t count,const int* center,const int* neighbor) {
    auto& atom=atoms_.at(a);
    if(atom.taken || p>=atom.pairs.size() || count>std::size_t(INT_MAX) ||
       (count && (!center || !neighbor)))
      throw std::invalid_argument("invalid overlap-list cache pair");
    auto left=storage<int>(count,allocate_),right=storage<int>(count,allocate_);
    if(count) {
      std::copy_n(center,count,left.get());
      std::copy_n(neighbor,count,right.get());
    }
    // Neither a short allocation nor a failed second allocation replaces a
    // previously stored pair. No shared counter is written by store workers.
    auto& pair=atom.pairs[p];
    pair.center=std::move(left);pair.neighbor=std::move(right);pair.ready=true;
    atom.counts[p]=int(count);
    atom.center[p]=pair.center.get();atom.neighbor[p]=pair.neighbor.get();
  }
  OpenMXOlgCacheAtom take_atom(std::size_t a) {
    auto& atom=atoms_.at(a);
    if(atom.taken || !complete(atom))
      throw std::invalid_argument("overlap-list cache atom is incomplete or already taken");
    OpenMXOlgCacheAtom result{atom.counts.data(),atom.center.get(),atom.neighbor.get(),atom.pairs.size()};
    for(auto& pair:atom.pairs) {(void)pair.center.release();(void)pair.neighbor.release();}
    (void)atom.center.release();(void)atom.neighbor.release();atom.taken=true;
    return result;
  }

 private:
  struct Free { template<class T> void operator()(T* p) const noexcept {std::free(p);} };
  template<class T> using Storage=std::unique_ptr<T[],Free>;
  template<class T> static Storage<T> storage(std::size_t count,Allocate allocate) {
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T))
      throw std::length_error("overlap-list allocation exceeds native address range");
    Storage<T> result(static_cast<T*>(allocate((count?count:1)*sizeof(T))));
    if(!result)throw std::bad_alloc();
    return result;
  }
  struct Pair { Storage<int> center,neighbor;bool ready=false; };
  struct Atom {
    int global;
    bool taken=false;
    std::vector<int> counts;
    std::vector<Pair> pairs;
    Storage<int*> center,neighbor;
    Atom(int g,std::size_t n,Allocate allocate)
        :global(g),counts(n),pairs(n),center(storage<int*>(n,allocate)),neighbor(storage<int*>(n,allocate)) {}
    Atom(Atom&&)=default;
    Atom& operator=(Atom&&)=default;
    Atom(const Atom&)=delete;
    Atom& operator=(const Atom&)=delete;
  };
  static bool complete(const Atom& atom) {
    return std::all_of(atom.pairs.begin(),atom.pairs.end(),[](const Pair& p){return p.ready;});
  }
  int md_iter_,copied_cells_;
  std::size_t global_atoms_;
  Allocate allocate_;
  std::vector<Atom> atoms_;
};
} // namespace openmx::core
