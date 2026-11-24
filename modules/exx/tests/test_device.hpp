// A device in host memory for the tests of the device path of the exchange
// build: the table OpenMXExxDevice of exx.h (section 11 of the implementation
// guide).  It does the work that the library gives it, lets one call fail on
// request, and holds the library to the contract of exx.h.  The first breach
// of the contract is kept as a text; a breach does not stop the work.
//
// The blocks of device memory come from malloc and are filled with the bytes
// 0xFF: a double that the library reads before it has written it is NaN, and
// an offset or a length is -1.  Every pointer that the library passes to
// dgemm, move, upload and download must lie in a live block together with
// everything that the call reads or writes there.
#pragma once

#include <openmx/exx/exx.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace exx_test {

class FakeDevice {
 public:
  // The calls after accept by kind.  All of them except release count as
  // calls that can fail.
  enum Kind { allocate_call, release_call, upload_call, download_call, dgemm_call, move_call, finish_call, kinds };

  FakeDevice() noexcept
      : table_{this,&accept_entry,&allocate_entry,&release_entry,&upload_entry,&download_entry,&dgemm_entry,
               &move_entry,&finish_entry}
  {
  }
  FakeDevice(const FakeDevice&) = delete;
  FakeDevice& operator=(const FakeDevice&) = delete;
  ~FakeDevice()
  {
    for (const auto& [start, size] : blocks_) std::free(reinterpret_cast<void*>(start));
  }

  const OpenMXExxDevice& table() const noexcept { return table_; }

  // Settings for the next call of the library: the answer of accept, and the
  // call that fails (counted from 0 after accept, release not counted), or
  // -1.  A failing allocate gives NULL; the other calls report 1 and do no
  // work.
  int answer = 1;
  long failing = -1;

  // Forgets the record of the calls before; blocks that were not released
  // stay.  The thread that calls reset() becomes the one that may call the
  // device.
  void reset()
  {
    accepted = 0;
    multiplications = 0.0;
    bytes = 0;
    allocated = 0;
    largest_product = 0.0;
    std::fill(std::begin(count),std::end(count),0L);
    calls = 0;
    sequence.clear();
    failed = false;
    breach.clear();
    pending_ = false;
    owner_ = std::this_thread::get_id();
  }

  // The record since reset().
  long accepted = 0;                 // calls of accept
  double multiplications = 0.0;      // the arguments of the last accept
  std::size_t bytes = 0;
  std::size_t allocated = 0;         // bytes of all allocations
  double largest_product = 0.0;      // the largest m n k of a dgemm
  long count[kinds] = {};            // calls of every kind
  long calls = 0;                    // calls that can fail
  std::vector<Kind> sequence;        // their kinds, in order
  bool failed = false;               // a call failed
  std::string breach;                // the first breach of the contract, empty without one

  std::size_t live() const noexcept { return blocks_.size(); }
  bool pending() const noexcept { return pending_; }

  // The record after a call of the library: no breach, every block released
  // and no work pending.  Empty when all is well.
  std::string ending() const
  {
    if (!breach.empty()) return breach;
    if (!blocks_.empty()) return "the library did not release all blocks of device memory";
    if (pending_) return "the library left work pending on the device";
    return "";
  }

 private:
  static FakeDevice& self(void* context) { return *static_cast<FakeDevice*>(context); }

  void note(const std::string& what)
  {
    if (breach.empty()) breach = what;
  }

  // A call of a kind: the checks that all calls share.  True when the call
  // is to fail.
  bool enter(Kind kind)
  {
    ++count[kind];
    if (std::this_thread::get_id() != owner_) note("the device is called from another thread than the library's caller");
    if (accepted != 1) note("the device is used without one call of accept before");
    if (failed && kind != finish_call && kind != release_call) note("the device is used after a failure");
    if (kind == release_call) return false;
    sequence.push_back(kind);
    if (calls++ != failing) return false;
    failed = true;
    return true;
  }

  // The live block that holds the bytes [address, address + size), as the
  // address of its first byte, or 0.  A range of no bytes needs an address
  // inside a block.  Addresses are integers, so that an address outside
  // every block can be formed and tested.
  static std::uintptr_t address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }
  std::uintptr_t block_of(std::uintptr_t at, std::size_t size) const
  {
    const auto after = blocks_.upper_bound(at);
    if (after == blocks_.begin()) return 0;
    const auto found = std::prev(after);
    const std::uintptr_t start = found->first;
    const std::size_t room = found->second;
    if (at-start >= room || size > room-(at-start)) return 0;
    return start;
  }
  std::uintptr_t block_of(const void* pointer, std::size_t size) const { return block_of(address(pointer),size); }

  bool inside(const void* pointer, std::size_t size, const std::string& what)
  {
    if (pointer != nullptr && block_of(pointer,size) != 0) return true;
    note(what+" lies outside the blocks of device memory");
    return false;
  }

  static int accept_entry(void* context, double multiplications, std::size_t bytes)
  {
    FakeDevice& d = self(context);
    if (std::this_thread::get_id() != d.owner_) d.note("accept is called from another thread than the library's caller");
    if (++d.accepted != 1 || d.calls != 0) d.note("accept is not the first call, or is called twice");
    d.multiplications = multiplications;
    d.bytes = bytes;
    return d.answer;
  }

  static void* allocate_entry(void* context, std::size_t bytes)
  {
    FakeDevice& d = self(context);
    if (d.enter(allocate_call)) return nullptr;
    void* pointer = std::malloc(std::max<std::size_t>(bytes,1));
    if (pointer == nullptr) {
      d.failed = true;
      return nullptr;
    }
    std::memset(pointer,0xFF,std::max<std::size_t>(bytes,1));
    d.blocks_.emplace(address(pointer),std::max<std::size_t>(bytes,1));
    d.allocated += bytes;
    return pointer;
  }

  static void release_entry(void* context, void* pointer)
  {
    FakeDevice& d = self(context);
    d.enter(release_call);
    if (d.pending_) d.note("release is called while work is pending");
    const auto found = d.blocks_.find(address(pointer));
    if (found == d.blocks_.end()) {
      d.note("release is called for a pointer that is not a live block");
      return;
    }
    std::free(pointer);
    d.blocks_.erase(found);
  }

  static int upload_entry(void* context, void* target, const void* source, std::size_t bytes)
  {
    FakeDevice& d = self(context);
    const bool fail = d.enter(upload_call);
    if (d.pending_) d.note("upload is called while work is pending");
    if (!d.inside(target,bytes,"the target of an upload") || source == nullptr || fail) return fail ? 1 : 2;
    std::memcpy(target,source,bytes);
    return 0;
  }

  static int download_entry(void* context, void* target, const void* source, std::size_t bytes)
  {
    FakeDevice& d = self(context);
    const bool fail = d.enter(download_call);
    if (d.pending_) d.note("download is called while work is pending");
    if (!d.inside(source,bytes,"the source of a download") || target == nullptr || fail) return fail ? 1 : 2;
    std::memcpy(target,source,bytes);
    return 0;
  }

  // The doubles of a column-major array of rows x columns with leading
  // dimension ld.
  static std::size_t extent(int rows, int columns, int ld)
  {
    if (rows <= 0 || columns <= 0) return 0;
    return static_cast<std::size_t>(columns-1)*static_cast<std::size_t>(ld)+static_cast<std::size_t>(rows);
  }

  static int dgemm_entry(void* context, char transa, char transb, int m, int n, int k, double alpha, const double* a,
                         int lda, const double* b, int ldb, double beta, double* c, int ldc)
  {
    FakeDevice& d = self(context);
    const bool fail = d.enter(dgemm_call);
    d.pending_ = true;
    const bool ta = transa == 'T', tb = transb == 'T';
    if ((!ta && transa != 'N') || (!tb && transb != 'N') || m < 0 || n < 0 || k < 0) {
      d.note("dgemm is called with arguments that DGEMM rejects");
      return 2;
    }
    const int a_rows = ta ? k : m, a_columns = ta ? m : k, b_rows = tb ? n : k, b_columns = tb ? k : n;
    if (lda < std::max(1,a_rows) || ldb < std::max(1,b_rows) || ldc < std::max(1,m)) {
      d.note("dgemm is called with a leading dimension that DGEMM rejects");
      return 2;
    }
    d.largest_product = std::max(d.largest_product,static_cast<double>(m)*static_cast<double>(n)*static_cast<double>(k));
    const std::size_t a_size = extent(a_rows,a_columns,lda), b_size = extent(b_rows,b_columns,ldb);
    const std::size_t c_size = extent(m,n,ldc);
    if ((a_size > 0 && !d.inside(a,a_size*sizeof(double),"the array A of a product")) ||
        (b_size > 0 && !d.inside(b,b_size*sizeof(double),"the array B of a product")) ||
        (c_size > 0 && !d.inside(c,c_size*sizeof(double),"the array C of a product")))
      return 2;
    if (c_size > 0 && ((a_size > 0 && d.block_of(c,1) == d.block_of(a,1)) ||
                       (b_size > 0 && d.block_of(c,1) == d.block_of(b,1))))
      d.note("the array C of a product shares a block with A or B");
    if (fail) return 1;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int l = 0; l < k; ++l)
          sum += (ta ? a[l+static_cast<std::size_t>(i)*lda] : a[i+static_cast<std::size_t>(l)*lda])*
                 (tb ? b[j+static_cast<std::size_t>(l)*ldb] : b[l+static_cast<std::size_t>(j)*ldb]);
        double& target = c[i+static_cast<std::size_t>(j)*ldc];
        target = alpha*sum+(beta == 0.0 ? 0.0 : beta*target);
      }
    return 0;
  }

  static long long offset(const OpenMXExxDeviceIndex& index, int r, int c)
  {
    return (index.rows != nullptr ? index.rows[r] : r*index.stride)+(index.columns != nullptr ? index.columns[c] : c*index.step);
  }

  static int move_entry(void* context, int rows, int columns, const int* lengths, double alpha, int add, double* target,
                        const OpenMXExxDeviceIndex* to, const double* source, const OpenMXExxDeviceIndex* from)
  {
    FakeDevice& d = self(context);
    const bool fail = d.enter(move_call);
    d.pending_ = true;
    if (rows < 0 || columns < 0 || (add != 0 && add != 1) || to == nullptr || from == nullptr) {
      d.note("move is called with invalid arguments");
      return 2;
    }
    const std::size_t r_count = static_cast<std::size_t>(rows), c_count = static_cast<std::size_t>(columns);
    for (const OpenMXExxDeviceIndex* index : {to,from}) {
      if ((index->rows != nullptr && !d.inside(index->rows,r_count*sizeof(long long),"a list of rows of a move")) ||
          (index->columns != nullptr && !d.inside(index->columns,c_count*sizeof(long long),"a list of columns of a move")))
        return 2;
    }
    if (lengths != nullptr) {
      if (!d.inside(lengths,r_count*sizeof(int),"the lengths of a move")) return 2;
      for (int r = 0; r < rows; ++r)
        if (lengths[r] < 0 || lengths[r] > columns) {
          d.note("a length of a move is negative or exceeds its columns");
          return 2;
        }
    }
    const std::uintptr_t t_block = d.block_of(target,sizeof(double));
    const std::uintptr_t s_block = d.block_of(source,sizeof(double));
    if (t_block == 0 || s_block == 0) {
      d.note("the target or the source of a move lies outside the blocks of device memory");
      return 2;
    }
    if (t_block == s_block) d.note("the target and the source of a move share a block");
    // The addresses of the elements as integers; an offset that leaves the
    // block of its array is noted before any element is touched.
    const auto at = [](const double* base, long long offset) {
      return address(base)+static_cast<std::uintptr_t>(offset)*sizeof(double);
    };
    std::vector<std::uintptr_t> targets;
    std::vector<std::uintptr_t> sources;
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < (lengths != nullptr ? lengths[r] : columns); ++c) {
        const std::uintptr_t t = at(target,offset(*to,r,c)), s = at(source,offset(*from,r,c));
        if (d.block_of(t,sizeof(double)) != t_block || d.block_of(s,sizeof(double)) != s_block) {
          d.note("an element of a move lies outside the block of its array");
          return 2;
        }
        targets.push_back(t);
        sources.push_back(s);
      }
    std::vector<std::uintptr_t> sorted = targets;
    std::sort(sorted.begin(),sorted.end());
    if (std::adjacent_find(sorted.begin(),sorted.end()) != sorted.end())
      d.note("two pairs of one move have the same target element");
    if (fail) return 1;
    for (std::size_t e = 0; e < targets.size(); ++e) {
      double& element = *reinterpret_cast<double*>(targets[e]);
      const double value = alpha**reinterpret_cast<const double*>(sources[e]);
      element = add != 0 ? element+value : value;
    }
    return 0;
  }

  static int finish_entry(void* context)
  {
    FakeDevice& d = self(context);
    const bool fail = d.enter(finish_call);
    d.pending_ = false;
    return fail ? 1 : 0;
  }

  OpenMXExxDevice table_;
  std::map<std::uintptr_t,std::size_t> blocks_;   // first byte and size of every live block
  bool pending_ = false;
  std::thread::id owner_ = std::this_thread::get_id();
};

} // namespace exx_test
