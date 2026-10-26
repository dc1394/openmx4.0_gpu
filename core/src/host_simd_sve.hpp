#ifndef OPENMX_HOST_SIMD_SVE_HPP
#define OPENMX_HOST_SIMD_SVE_HPP

// Private Kokkos SIMD ABI backed by ARM ACLE. Kokkos 5.2.2's SVE ABI lacks
// the half-width float conversions used by OpenMX and rejects SVE-512's
// half-width integer indices. Keep both full and half vectors in SVE registers,
// with predicates limiting loads/stores to their logical lane count.
// Bits is part of every type: different VL objects must never share template
// instantiations, including in unoptimized builds. No SVE type crosses the
// host library's POD boundary. Dispatch checks the calling thread's VL.
#include <Kokkos_SIMD_Common.hpp>
#include <arm_sve.h>
#include <iterator>

#ifndef __ARM_FEATURE_SVE_BITS
#error "The private SVE ABI requires a fixed -msve-vector-bits value"
#endif

namespace Kokkos::Experimental {
namespace simd_abi {
template<int N, int Bits> struct openmx_sve_fixed_size {};
}
namespace Impl {
template<class T, int Bits> struct OpenMXSVEStorage;
#define OPENMX_SVE_STORAGE(T, SVE) \
  template<int Bits> struct OpenMXSVEStorage<T, Bits> { \
    static_assert(Bits == __ARM_FEATURE_SVE_BITS); \
    typedef SVE type __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS))); \
  };
OPENMX_SVE_STORAGE(double, svfloat64_t)
OPENMX_SVE_STORAGE(float, svfloat32_t)
OPENMX_SVE_STORAGE(std::int32_t, svint32_t)
#undef OPENMX_SVE_STORAGE
template<class V> concept OpenMXSVEVector = requires { V::openmx_sve; };
}

template<class T, int N, int Bits>
class basic_simd_mask<T, simd_abi::openmx_sve_fixed_size<N, Bits>> {
  typedef svbool_t Native __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
  Native value_;
public:
  using value_type = bool;
  using abi_type = simd_abi::openmx_sve_fixed_size<N, Bits>;
  static constexpr std::integral_constant<int, N> size{};
  static svbool_t active() noexcept {
    if constexpr(sizeof(T) == 8) return svwhilelt_b64(0, N);
    else return svwhilelt_b32(0, N);
  }
  static basic_simd_mask from_native(svbool_t value) noexcept {
    basic_simd_mask result;
    result.value_ = svand_b_z(active(), value, active());
    return result;
  }
  svbool_t native() const noexcept { return value_; }
  basic_simd_mask() noexcept = default;
  explicit basic_simd_mask(bool value) noexcept : value_(value ? active() : svpfalse_b()) {}
  template<class G> requires requires(G generate) { { generate(0) } -> std::convertible_to<bool>; }
  explicit basic_simd_mask(G&& generate) noexcept {
    using Lane = std::conditional_t<sizeof(T) == 8, std::int64_t, std::int32_t>;
    Lane values[N];
    for(int i=0; i<N; ++i) values[i] = generate(i) ? 1 : 0;
    value_ = svcmpne(active(), svld1(active(), values), 0);
  }
  template<class U>
  basic_simd_mask(const basic_simd_mask<U, abi_type>& other) noexcept
      : basic_simd_mask([&](int i) { return other[i]; }) {}
  bool operator[](int i) const noexcept {
    if constexpr(sizeof(T) == 8) return svptest_last(svwhilele_b64(0, i), value_);
    else return svptest_last(svwhilele_b32(0, i), value_);
  }
  basic_simd_mask operator!() const noexcept { return from_native(svnot_b_z(active(), value_)); }
  basic_simd_mask operator~() const noexcept { return !*this; }
  friend basic_simd_mask operator&&(basic_simd_mask a, basic_simd_mask b) noexcept {
    return from_native(svand_b_z(active(), a.value_, b.value_));
  }
  friend basic_simd_mask operator||(basic_simd_mask a, basic_simd_mask b) noexcept {
    return from_native(svorr_b_z(active(), a.value_, b.value_));
  }
  friend basic_simd_mask operator&(basic_simd_mask a, basic_simd_mask b) noexcept { return a && b; }
  friend basic_simd_mask operator|(basic_simd_mask a, basic_simd_mask b) noexcept { return a || b; }
  friend basic_simd_mask operator^(basic_simd_mask a, basic_simd_mask b) noexcept {
    return from_native(sveor_b_z(active(), a.value_, b.value_));
  }
  basic_simd_mask& operator&=(basic_simd_mask b) noexcept { return *this = *this & b; }
  basic_simd_mask& operator|=(basic_simd_mask b) noexcept { return *this = *this | b; }
};

template<class T, int N, int Bits>
class basic_simd<T, simd_abi::openmx_sve_fixed_size<N, Bits>> {
  using Native = typename Impl::OpenMXSVEStorage<T, Bits>::type;
  Native value_;
public:
  static_assert(N > 0 && N * sizeof(T) * 8 <= Bits);
  static constexpr bool openmx_sve = true;
  static constexpr std::integral_constant<int, N> size{};
  using value_type = T;
  using abi_type = simd_abi::openmx_sve_fixed_size<N, Bits>;
  using mask_type = basic_simd_mask<T, abi_type>;
  static svbool_t active() noexcept { return mask_type::active(); }
  Native native() const noexcept { return value_; }
  basic_simd() noexcept = default;
  explicit basic_simd(Native value) noexcept : value_(value) {}
  template<class U> requires std::is_convertible_v<U, T>
  basic_simd(U value) noexcept {
    if constexpr(std::is_same_v<T, double>) value_ = svdup_f64(T(value));
    else if constexpr(std::is_same_v<T, float>) value_ = svdup_f32(T(value));
    else value_ = svdup_s32(T(value));
  }
  template<class G> requires requires(G generate) { { generate(0) } -> std::convertible_to<T>; }
  explicit basic_simd(G&& generate) noexcept {
    T values[N];
    for(int i=0; i<N; ++i) values[i] = T(generate(i));
    value_ = svld1(active(), values);
  }
  template<class U>
  explicit basic_simd(const basic_simd<U, abi_type>& other) noexcept {
    if constexpr(std::is_same_v<T, double> && std::is_same_v<U, float>)
      value_ = svcvt_f64_x(active(), svzip1_f32(other.native(), other.native()));
    else if constexpr(std::is_same_v<T, float> && std::is_same_v<U, double>) {
      const auto converted = svcvt_f32_x(other.active(), other.native());
      value_ = svuzp1_f32(converted, converted);
    } else if constexpr(std::is_same_v<T, double> && std::is_same_v<U, std::int32_t>)
      value_ = svcvt_f64_x(active(), svunpklo_s64(other.native()));
    else if constexpr(std::is_same_v<T, std::int32_t> && std::is_same_v<U, double>) {
      const auto converted = svreinterpret_s32(svcvt_s64_x(other.active(), other.native()));
      value_ = svuzp1_s32(converted, converted);
    } else {
      T values[N];
      for(int i=0; i<N; ++i) values[i] = T(other[i]);
      value_ = svld1(active(), values);
    }
  }
  template<class... Flags>
  explicit basic_simd(const T* p, simd_flags<Flags...> = {}) noexcept : value_(svld1(active(), p)) {}
  T operator[](int i) const noexcept {
    if constexpr(sizeof(T) == 8) return svlastb(svwhilele_b64(0, i), value_);
    else return svlastb(svwhilele_b32(0, i), value_);
  }
  basic_simd operator-() const noexcept { return basic_simd(svneg_x(active(), value_)); }
#define OPENMX_SVE_BINARY(OP, INTRINSIC) \
  friend basic_simd operator OP(basic_simd a, basic_simd b) noexcept { \
    return basic_simd(INTRINSIC(active(), a.value_, b.value_)); \
  }
  OPENMX_SVE_BINARY(+, svadd_x)
  OPENMX_SVE_BINARY(-, svsub_x)
  OPENMX_SVE_BINARY(*, svmul_x)
  OPENMX_SVE_BINARY(/, svdiv_x)
#undef OPENMX_SVE_BINARY
#define OPENMX_SVE_COMPARE(OP, INTRINSIC) \
  friend mask_type operator OP(basic_simd a, basic_simd b) noexcept { \
    return mask_type::from_native(INTRINSIC(active(), a.value_, b.value_)); \
  }
  OPENMX_SVE_COMPARE(==, svcmpeq)
  OPENMX_SVE_COMPARE(!=, svcmpne)
  OPENMX_SVE_COMPARE(<, svcmplt)
  OPENMX_SVE_COMPARE(<=, svcmple)
  OPENMX_SVE_COMPARE(>, svcmpgt)
  OPENMX_SVE_COMPARE(>=, svcmpge)
#undef OPENMX_SVE_COMPARE
};

template<Impl::OpenMXSVEVector V, class... Flags>
V simd_unchecked_load(const typename V::value_type* p, simd_flags<Flags...> = {}) noexcept {
  return V(svld1(V::active(), p));
}
template<Impl::OpenMXSVEVector V, class... Flags>
V simd_partial_load(const typename V::value_type* p, const typename V::mask_type& mask,
                    simd_flags<Flags...> = {}) noexcept {
  return V(svld1(mask.native(), p));
}
template<Impl::OpenMXSVEVector V, class... Flags>
V simd_unchecked_load(const typename V::value_type* p, const typename V::mask_type& mask,
                      simd_flags<Flags...> flags = {}) noexcept {
  return simd_partial_load<V>(p, mask, flags);
}
template<Impl::OpenMXSVEVector V, class... Flags>
void simd_unchecked_store(V value, typename V::value_type* p, simd_flags<Flags...> = {}) noexcept {
  svst1(V::active(), p, value.native());
}
template<Impl::OpenMXSVEVector V, class... Flags>
void simd_partial_store(V value, typename V::value_type* p, const typename V::mask_type& mask,
                        simd_flags<Flags...> = {}) noexcept {
  svst1(mask.native(), p, value.native());
}
template<Impl::OpenMXSVEVector V, class... Flags>
void simd_unchecked_store(V value, typename V::value_type* p, const typename V::mask_type& mask,
                          simd_flags<Flags...> flags = {}) noexcept {
  simd_partial_store(value, p, mask, flags);
}
template<Impl::OpenMXSVEVector V, class M>
V condition(const M& mask, V yes, V no) noexcept {
  return V(svsel(typename V::mask_type(mask).native(), yes.native(), no.native()));
}
template<class T, int N, int Bits>
bool any_of(const basic_simd_mask<T, simd_abi::openmx_sve_fixed_size<N, Bits>>& mask) noexcept {
  return svptest_any(mask.active(), mask.native());
}
template<class T, int N, int Bits>
bool all_of(const basic_simd_mask<T, simd_abi::openmx_sve_fixed_size<N, Bits>>& mask) noexcept {
  return !any_of(!mask);
}
template<class T, int N, int Bits>
bool none_of(const basic_simd_mask<T, simd_abi::openmx_sve_fixed_size<N, Bits>>& mask) noexcept {
  return !any_of(mask);
}
template<Impl::OpenMXSVEVector V, class R, class M, Impl::OpenMXSVEVector I, class... Flags>
V unchecked_gather_from(const R& range, const M& mask, I index, simd_flags<Flags...> = {}) noexcept {
  const typename V::mask_type active(mask);
  if constexpr(sizeof(typename V::value_type) == 8)
    return V(svld1_gather_index(active.native(), std::data(range), svunpklo_s64(index.native())));
  else
    return V(svld1_gather_index(active.native(), std::data(range), index.native()));
}
template<Impl::OpenMXSVEVector V, class R, Impl::OpenMXSVEVector I, class... Flags>
V unchecked_gather_from(const R& range, I index, simd_flags<Flags...> flags = {}) noexcept {
  return unchecked_gather_from<V>(range, typename V::mask_type(true), index, flags);
}
template<Impl::OpenMXSVEVector V, class R, class M, Impl::OpenMXSVEVector I, class... Flags>
void unchecked_scatter_to(V value, const R& range, const M& mask, I index,
                          simd_flags<Flags...> = {}) noexcept {
  const typename V::mask_type active(mask);
  if constexpr(sizeof(typename V::value_type) == 8)
    svst1_scatter_index(active.native(), std::data(range), svunpklo_s64(index.native()), value.native());
  else
    svst1_scatter_index(active.native(), std::data(range), index.native(), value.native());
}
template<Impl::OpenMXSVEVector V, class R, Impl::OpenMXSVEVector I, class... Flags>
void unchecked_scatter_to(V value, const R& range, I index, simd_flags<Flags...> flags = {}) noexcept {
  unchecked_scatter_to(value, range, typename V::mask_type(true), index, flags);
}
}

namespace Kokkos {
#define OPENMX_SVE_MATH(NAME, INTRINSIC) \
  template<class T, int N, int Bits> requires std::is_floating_point_v<T> \
  Experimental::basic_simd<T, Experimental::simd_abi::openmx_sve_fixed_size<N, Bits>> \
  NAME(const Experimental::basic_simd<T, Experimental::simd_abi::openmx_sve_fixed_size<N, Bits>>& value) { \
    using V = Experimental::basic_simd<T, Experimental::simd_abi::openmx_sve_fixed_size<N, Bits>>; \
    return V(INTRINSIC(V::active(), value.native())); \
  }
OPENMX_SVE_MATH(sqrt, svsqrt_x)
OPENMX_SVE_MATH(floor, svrintm_x)
OPENMX_SVE_MATH(abs, svabs_x)
#undef OPENMX_SVE_MATH
}
#endif
