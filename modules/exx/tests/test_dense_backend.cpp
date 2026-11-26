// Test of dense_backend.hpp:
//   exx-dense-backend  the built-in product against sums written out, for
//                      every transposition, padded arrays and special
//                      factors; the three row-major products with and
//                      without a table; failures of a table; the loop of a
//                      table, on one thread and on several, the number of
//                      calls that it is asked for, and loops that break the
//                      contract; the loops of the tests themselves when a
//                      thread cannot be created and when a loop is called
//                      from a body or from a thread that does not own its
//                      context
#include <openmx/exx/dense_backend.hpp>

#include "test_dgemm.hpp"
#include "test_parallel.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace exx_test {

using namespace openmx::exx;

namespace {

using Product = int (*)(void*, char, char, int, int, int, double, const double*, int, const double*, int, double,
                        double*, int);

struct Case {
  char transa, transb;
  int m, n, k;
  double alpha, beta;
  int padding; // rows added to every leading dimension
};

bool transposes(char letter) { return letter != 'N' && letter != 'n'; }

bool same_bits(double a, double b) { return std::memcmp(&a,&b,sizeof a) == 0; }

// One call with column-major arrays.  The padding of A and B holds NaN, so a
// product that reads it gives NaN; the padding of C holds a sentinel that must
// survive.  With beta = 0 the elements of C hold NaN on entry and must not be
// read.
void check_product(Product product, const Case& c, Numbers& random, Worst& worst)
{
  const double nan = std::numeric_limits<double>::quiet_NaN(), sentinel = 12345.678;
  const bool ta = transposes(c.transa), tb = transposes(c.transb);
  const std::size_t m = static_cast<std::size_t>(c.m), n = static_cast<std::size_t>(c.n), k = static_cast<std::size_t>(c.k);
  const std::size_t rows_a = ta ? k : m, columns_a = ta ? m : k, rows_b = tb ? n : k, columns_b = tb ? k : n;
  const std::size_t pad = static_cast<std::size_t>(c.padding);
  const std::size_t lda = std::max<std::size_t>(rows_a,1)+pad, ldb = std::max<std::size_t>(rows_b,1)+pad;
  const std::size_t ldc = std::max<std::size_t>(m,1)+pad;
  std::vector<double> a(lda*columns_a,nan), b(ldb*columns_b,nan), result(ldc*n,sentinel);
  for (std::size_t j = 0; j < columns_a; ++j)
    for (std::size_t i = 0; i < rows_a; ++i) a[i+j*lda] = random.next();
  for (std::size_t j = 0; j < columns_b; ++j)
    for (std::size_t i = 0; i < rows_b; ++i) b[i+j*ldb] = random.next();
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < m; ++i) result[i+j*ldc] = c.beta == 0.0 ? nan : random.next();
  const std::vector<double> before = result;

  const int status = product(nullptr,c.transa,c.transb,c.m,c.n,c.k,c.alpha,a.data(),static_cast<int>(lda),b.data(),
                             static_cast<int>(ldb),c.beta,result.data(),static_cast<int>(ldc));
  require(status == 0,"a valid product reports a failure");
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < m; ++i) {
      double sum = 0.0;
      for (std::size_t l = 0; l < k; ++l) sum += (ta ? a[l+i*lda] : a[i+l*lda])*(tb ? b[j+l*ldb] : b[l+j*ldb]);
      worst.absolute(result[i+j*ldc],c.alpha*sum+(c.beta == 0.0 ? 0.0 : c.beta*before[i+j*ldc]));
    }
    for (std::size_t i = m; i < ldc; ++i)
      require(same_bits(result[i+j*ldc],sentinel),"a product writes into the padding of C");
  }
}

// The three row-major products of a backend against sums written out.
void check_row_major(const DenseBackend& backend, std::size_t m, std::size_t n, std::size_t k, double alpha, double beta,
                     Numbers& random, Worst& worst)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto filled = [&random](std::size_t count) {
    std::vector<double> values(count);
    for (double& value : values) value = random.next();
    return values;
  };
  const auto start = [&](std::size_t count) { return beta == 0.0 ? std::vector<double>(count,nan) : filled(count); };
  const auto old = [beta](const std::vector<double>& before, std::size_t e) { return beta == 0.0 ? 0.0 : beta*before[e]; };

  {  // C = alpha A B + beta C
    const std::vector<double> a = filled(m*k), b = filled(k*n), before = start(m*n);
    std::vector<double> c = before;
    backend.multiply(m,n,k,alpha,a.data(),b.data(),beta,c.data());
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (std::size_t l = 0; l < k; ++l) sum += a[i*k+l]*b[l*n+j];
        worst.absolute(c[i*n+j],alpha*sum+old(before,i*n+j));
      }
  }
  {  // C = alpha A B^T + beta C, B is n x k
    const std::vector<double> a = filled(m*k), b = filled(n*k), before = start(m*n);
    std::vector<double> c = before;
    backend.multiply_transposed(m,n,k,alpha,a.data(),b.data(),beta,c.data());
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (std::size_t l = 0; l < k; ++l) sum += a[i*k+l]*b[j*k+l];
        worst.absolute(c[i*n+j],alpha*sum+old(before,i*n+j));
      }
  }
  {  // C = alpha A^T B + beta C, A is k x m
    const std::vector<double> a = filled(k*m), b = filled(k*n), before = start(m*n);
    std::vector<double> c = before;
    backend.transposed_multiply(m,n,k,alpha,a.data(),b.data(),beta,c.data());
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (std::size_t l = 0; l < k; ++l) sum += a[l*m+i]*b[l*n+j];
        worst.absolute(c[i*n+j],alpha*sum+old(before,i*n+j));
      }
  }
}

template <class Exception, class Call>
void expect(const std::string& what, Call call)
{
  bool thrown = false;
  try { call(); } catch (const Exception&) { thrown = true; }
  require(thrown,"no exception for "+what);
}

// The indices in the order in which a loop on one thread calls them.
std::vector<std::size_t> order_of(const DenseBackend& backend, std::size_t count)
{
  std::vector<std::size_t> order;
  backend.parallel_for(count,[&order](std::size_t index) { order.push_back(index); });
  return order;
}

// DenseBackend::parallel_for() with the loops of test_parallel.hpp.
void check_loops(Numbers& random)
{
  const std::vector<std::size_t> forward{0,1,2,3,4}, backward{4,3,2,1,0};
  Calls calls;
  // The backend keeps a copy of the table.
  const auto with = [&calls](Loop loop, int threads) {
    const OpenMXExxDenseBackend table{nullptr,&calls,loop,threads,nullptr};
    return DenseBackend(&table);
  };

  // Without a loop, and with fewer than two threads, the calls run in order
  // on the calling thread.
  require(DenseBackend().pieces() == 1 && order_of(DenseBackend(),5) == forward,"wrong loop without a table");
  for (const int threads : {1,0,-3}) {
    const DenseBackend backend = with(&reversed_loop,threads);
    require(backend.pieces() == 1 && order_of(backend,5) == forward,"a table with fewer than two threads runs its loop");
  }
  {
    const DenseBackend backend = with(nullptr,4);
    require(backend.pieces() == 1 && order_of(backend,5) == forward,"threads without a loop change the loop");
  }
  require(calls.loops == 0,"a loop of a table was called without need");

  // The loop of the table makes the calls; the number of calls is the
  // caller's, not the number of threads.  A table without dgemm keeps the
  // built-in products.
  {
    const DenseBackend backend = with(&reversed_loop,2);
    require(backend.pieces() == 2 && backend.built_in(),"wrong report of a table with a loop only");
    require(with(&in_order_loop,7).pieces() == 7 && with(&threaded_loop,64).pieces() == 64,
            "the pieces of a table are not its threads");
    require(order_of(backend,5) == backward && calls.loops == 1,"the loop of the table does not make the calls");
    require(order_of(with(&in_order_loop,7),5) == forward && calls.loops == 2,"wrong calls of a loop in order");
    // The count that the two loops were asked for.  A loop of the tests makes
    // the calls that it is asked for, so another count already ends one of
    // the two lines above, with the exception of the library for a call that
    // is missing or does not exist; this line names the count itself.
    require(calls.counts == Counts({{5,2}}),"the loop of the table is not asked for the calls of the caller");
    // Nothing to divide: no loop of the table.
    require(order_of(backend,0).empty() && order_of(backend,1) == std::vector<std::size_t>{0} && calls.loops == 2,
            "a loop with fewer than two calls reaches the table");
    Worst worst("row-major products, table with a loop only",1e-13);
    check_row_major(backend,3,4,5,0.7,0.3,random,worst);
    worst.finish();
  }

  // An exception of a call: the other calls of a loop of the table still
  // run, and the exception of the lowest index is thrown again.  The lowest
  // call that throws is the first call of the loop, the second or the last,
  // and every call after it throws an exception of another type; with the
  // reversed loop those are the first exceptions in time.  On the calling
  // thread the loop ends at once.
  for (const std::size_t lowest : {std::size_t{0},std::size_t{1},std::size_t{4}})
    for (const Loop loop : {&in_order_loop,&reversed_loop,&threaded_loop}) {
      const DenseBackend backend = with(loop,3);
      std::array<std::atomic<int>,5> ran{};
      expect<std::invalid_argument>("a call that throws",[&] {
        backend.parallel_for(5,[&ran,lowest](std::size_t index) {
          ++ran[index];
          if (index == lowest) throw std::invalid_argument("the lowest call that throws");
          if (index > lowest)
            throw std::out_of_range("a loop throws the exception of a later call, not that of the lowest index");
        });
      });
      require(std::all_of(ran.begin(),ran.end(),[](const std::atomic<int>& count) { return count == 1; }),
              "an exception of one call keeps another call from running");
    }
  {
    std::vector<std::size_t> ran;
    expect<std::out_of_range>("a call that throws on the calling thread",[&] {
      DenseBackend().parallel_for(5,[&ran](std::size_t index) {
        ran.push_back(index);
        if (index == 1) throw std::out_of_range("the second call");
      });
    });
    require(ran == std::vector<std::size_t>({0,1}),"the loop on the calling thread goes on after an exception");
  }

  // Loops that break the contract.  A call made twice, or with an index that
  // does not exist, does not reach the body.  The loop that cannot create the
  // thread of its last call reports a failure after the other calls have run
  // on their threads; it joins them, or the program would end here.
  for (const Loop loop : {&failing_loop,&skipping_loop,&repeating_loop,&stray_loop,&starved_loop}) {
    const DenseBackend backend = with(loop,2);
    std::array<int,5> ran{};
    expect<std::runtime_error>("a loop that breaks the contract",[&] {
      backend.parallel_for(5,[&ran](std::size_t index) { ++ran.at(index); });
    });
    const int last = loop == &skipping_loop || loop == &starved_loop ? 0 : 1;
    require(ran == std::array<int,5>({1,1,1,1,last}),"a loop that breaks the contract runs a call twice or a wrong one");
  }
  // The loop that cannot create a thread reports that itself.  The check
  // above does not show it: the library ends a loop that leaves out a call
  // with the same exception, whatever the loop reports.
  {
    std::atomic<int> ran{0};
    const auto body = [](void* counted, int) { ++*static_cast<std::atomic<int>*>(counted); };
    require(starved_loop(nullptr,5,body,&ran) != 0 && ran == 4,
            "a loop that cannot create a thread reports no failure, or does not run the other calls");
  }
  // The library promises not to call a loop from a body (exx.h), and the
  // loops of the tests hold it to that: a loop that is called from one of
  // their calls makes no call and reports a failure.
  for (const Loop loop : {&in_order_loop,&reversed_loop,&threaded_loop}) {
    struct Nested {
      Loop loop;
      std::atomic<int> refused{0}, ran{0};
    } nested{loop};
    const int status = loop(nullptr,3,[](void* argument, int) {
      Nested& outer = *static_cast<Nested*>(argument);
      const int inner = outer.loop(nullptr,2,[](void* counted, int) { ++static_cast<Nested*>(counted)->ran; },&outer);
      if (inner != 0) ++outer.refused;
    },&nested);
    require(status == 0 && nested.refused == 3 && nested.ran == 0,"a loop of the tests accepts a call from a body");
  }
  // The same for a loop that is called from another thread than the one
  // that made its context: the counts of the context have no lock.
  for (const Loop loop : {&in_order_loop,&reversed_loop,&threaded_loop}) {
    Calls mine;
    std::atomic<int> ran{0};
    int status = 0;
    const auto body = [](void* counted, int) { ++*static_cast<std::atomic<int>*>(counted); };
    std::thread([&] { status = loop(&mine,2,body,&ran); }).join();
    require(status != 0 && ran == 0 && mine.loops == 0 && mine.counts.empty(),
            "a loop of the tests accepts a call from a thread that does not own its context");
    require(loop(&mine,2,body,&ran) == 0 && ran == 2 && mine.counts == Counts({{2,1}}),
            "a loop of the tests refuses the thread that owns its context");
  }
  {
    // The exception of a call comes before the failure of the loop.
    const DenseBackend backend = with(&failing_loop,2);
    expect<std::invalid_argument>("a call that throws in a loop that fails",[&] {
      backend.parallel_for(3,[](std::size_t index) { if (index == 2) throw std::invalid_argument("the last call"); });
    });
    expect<std::length_error>("a loop with more calls than the int of the table",[&] {
      backend.parallel_for(static_cast<std::size_t>(INT_MAX)+1,[](std::size_t) {});
    });
  }

  // Calls on threads of their own, each with products of its own: the
  // built-in loops and a product of a table.
  for (const bool table : {false,true}) {
    const OpenMXExxDenseBackend threaded{table ? &shared_dgemm : nullptr,&calls,&threaded_loop,8,nullptr};
    const DenseBackend backend(&threaded);
    const std::size_t m = 5, n = 4, k = 6, count = 8;
    std::vector<double> a(count*m*k), b(count*k*n), c(count*m*n,0.0), expected(count*m*n,0.0);
    for (double& value : a) value = random.next();
    for (double& value : b) value = random.next();
    for (std::size_t index = 0; index < count; ++index)
      backend.multiply(m,n,k,1.0,a.data()+index*m*k,b.data()+index*k*n,0.0,expected.data()+index*m*n);
    const long before = calls.products;
    backend.parallel_for(count,[&](std::size_t index) {
      backend.multiply(m,n,k,1.0,a.data()+index*m*k,b.data()+index*k*n,0.0,c.data()+index*m*n);
    });
    require(c == expected,"products on several threads differ from the same products on one thread");
    // Only a table with a product counts products.
    if (table)
      require(calls.products-before == static_cast<long>(count),"wrong number of products on several threads");
  }
}

} // namespace

void dense_backend()
{
  Numbers random;
  const std::vector<std::array<int,3>> sizes{{3,4,5},{1,1,1},{5,2,7},{4,3,0},{2,6,1}};
  const std::vector<std::array<double,2>> factors{{1.0,0.0},{-0.5,1.0},{0.7,0.3},{0.0,0.3},{0.0,0.0},{1.3,1.0}};

  // The built-in product, and the second implementation that the other tests
  // pass as a table.
  const Product products[] = {&openmx_exx_reference_dgemm,&counting_dgemm};
  for (const Product product : products) {
    Worst worst(product == &openmx_exx_reference_dgemm ? "the built-in product against sums written out"
                                                       : "the product of the tests against sums written out",1e-13);
    for (const char transa : {'N','T'})
      for (const char transb : {'N','T'})
        for (const std::array<int,3>& size : sizes)
          for (const std::array<double,2>& factor : factors)
            for (const int padding : {0,2})
              check_product(product,{transa,transb,size[0],size[1],size[2],factor[0],factor[1],padding},random,worst);
    check_product(product,{'n','t',3,2,4,0.9,0.1,1},random,worst);
    check_product(product,{'t','n',3,2,4,0.9,0.1,1},random,worst);
    worst.finish();
  }
  // For real matrices the conjugate transpose is the transpose.
  {
    Worst worst("the built-in product with 'C'",1e-13);
    check_product(&openmx_exx_reference_dgemm,{'C','N',3,4,2,1.1,0.4,1},random,worst);
    check_product(&openmx_exx_reference_dgemm,{'N','c',3,4,2,1.1,0.4,1},random,worst);
    worst.finish();
  }

  // Nothing to compute: C keeps its values, and the other arrays are not read.
  {
    double c[4] = {1.0,2.0,3.0,4.0};
    require(openmx_exx_reference_dgemm(nullptr,'N','N',0,5,3,1.0,nullptr,1,nullptr,3,0.0,nullptr,1) == 0,
            "a product without rows fails");
    require(openmx_exx_reference_dgemm(nullptr,'N','N',5,0,3,1.0,nullptr,5,nullptr,3,0.0,nullptr,5) == 0,
            "a product without columns fails");
    require(openmx_exx_reference_dgemm(nullptr,'N','N',2,2,0,1.0,nullptr,2,nullptr,1,1.0,c,2) == 0 && c[0] == 1.0 &&
            c[3] == 4.0,"an empty sum with beta = 1 changes C");
    require(openmx_exx_reference_dgemm(nullptr,'N','N',2,2,0,1.0,nullptr,2,nullptr,1,0.0,c,2) == 0 && c[0] == 0.0 &&
            c[3] == 0.0,"an empty sum with beta = 0 does not clear C");
  }

  // Arguments that DGEMM rejects: the status is 1 and C is untouched.
  {
    const std::vector<double> a(6*6,1.0), b(6*6,1.0);
    std::vector<double> c(6*6,2.5);
    const auto rejected = [&](const std::string& what, char transa, char transb, int m, int n, int k, const double* pa,
                              int lda, const double* pb, int ldb, double* pc, int ldc) {
      require(openmx_exx_reference_dgemm(nullptr,transa,transb,m,n,k,1.0,pa,lda,pb,ldb,0.0,pc,ldc) == 1,
              "the built-in product accepts "+what);
      require(std::all_of(c.begin(),c.end(),[](double value) { return value == 2.5; }),
              "the built-in product changes C after rejecting "+what);
    };
    rejected("an unknown letter for A",'X','N',2,2,2,a.data(),2,b.data(),2,c.data(),2);
    rejected("an unknown letter for B",'N','?',2,2,2,a.data(),2,b.data(),2,c.data(),2);
    rejected("a negative number of rows",'N','N',-1,2,2,a.data(),2,b.data(),2,c.data(),2);
    rejected("a negative number of columns",'N','N',2,-1,2,a.data(),2,b.data(),2,c.data(),2);
    rejected("a negative inner dimension",'N','N',2,2,-1,a.data(),2,b.data(),2,c.data(),2);
    rejected("a leading dimension of A below m",'N','N',3,2,2,a.data(),2,b.data(),2,c.data(),3);
    rejected("a leading dimension of A below k for A^T",'T','N',2,2,3,a.data(),2,b.data(),3,c.data(),2);
    rejected("a leading dimension of B below k",'N','N',2,2,3,a.data(),2,b.data(),2,c.data(),2);
    rejected("a leading dimension of B below n for B^T",'N','T',2,3,2,a.data(),2,b.data(),2,c.data(),2);
    rejected("a leading dimension of C below m",'N','N',3,2,2,a.data(),3,b.data(),2,c.data(),2);
    rejected("a leading dimension of zero",'N','N',0,2,0,a.data(),0,b.data(),1,c.data(),1);
    rejected("a missing A",'N','N',2,2,2,nullptr,2,b.data(),2,c.data(),2);
    rejected("a missing B",'N','N',2,2,2,a.data(),2,nullptr,2,c.data(),2);
    rejected("a missing C",'N','N',2,2,2,a.data(),2,b.data(),2,nullptr,2);
  }

  // The row-major products, with the built-in loops and with a table.
  long calls = 0;
  const OpenMXExxDenseBackend table{&counting_dgemm,&calls,nullptr,0,nullptr};
  const DenseBackend built_in, with_table(&table);
  require(built_in.built_in() && !with_table.built_in(),"wrong report of where the products come from");
  require(DenseBackend(nullptr).built_in(),"a missing table does not select the built-in loops");
  {
    const OpenMXExxDenseBackend without_function{nullptr,&calls,nullptr,0,nullptr};
    require(DenseBackend(&without_function).built_in(),"a table without a function does not select the built-in loops");
  }
  for (const DenseBackend* backend : {&built_in,&with_table}) {
    Worst worst(backend->built_in() ? "row-major products, built-in loops" : "row-major products, table",1e-13);
    for (const std::array<int,3>& size : sizes)
      for (const std::array<double,2>& factor : factors)
        check_row_major(*backend,static_cast<std::size_t>(size[0]),static_cast<std::size_t>(size[1]),
                        static_cast<std::size_t>(size[2]),factor[0],factor[1],random,worst);
    worst.finish();
  }
  // Three products for each of the 5 x 6 cases went through the table; the
  // built-in backend did not touch it.
  require(calls == 3*5*6,"the table was not called once for every product");
  {
    // Without rows or columns there is no product, and nothing is called.
    double value = 3.0;
    with_table.multiply(0,4,2,1.0,nullptr,nullptr,0.0,&value);
    with_table.multiply_transposed(3,0,2,1.0,nullptr,nullptr,0.0,&value);
    with_table.transposed_multiply(0,0,2,1.0,nullptr,nullptr,0.0,&value);
    require(calls == 3*5*6 && value == 3.0,"an empty product calls the table");
  }

  // A table that fails, and sizes beyond the int of the table.
  {
    const OpenMXExxDenseBackend failing{&failing_dgemm,nullptr,nullptr,0,nullptr};
    const DenseBackend backend(&failing);
    const double a = 1.0, b = 2.0;
    double c = 0.0;
    expect<std::runtime_error>("a table that reports a failure",[&] { backend.multiply(1,1,1,1.0,&a,&b,0.0,&c); });
    expect<std::runtime_error>("an invalid letter",[&] { built_in.gemm('X','N',1,1,1,1.0,&a,1,&b,1,0.0,&c,1); });
    const std::size_t huge = static_cast<std::size_t>(INT_MAX)+1;
    expect<std::length_error>("a matrix with too many rows",[&] { built_in.gemm('N','N',huge,1,1,1.0,&a,huge,&b,1,0.0,&c,huge); });
    expect<std::length_error>("a matrix with too many columns",[&] { with_table.gemm('N','N',1,huge,1,1.0,&a,1,&b,1,0.0,&c,1); });
    require(calls == 3*5*6,"a product that is too large reaches the table");
  }

  check_loops(random);
}

} // namespace exx_test
