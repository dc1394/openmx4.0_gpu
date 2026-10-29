#ifndef OPENMX_IO_BASIS_SPEC_HPP
#define OPENMX_IO_BASIS_SPEC_HPP

// The grammar of OpenMX basis specifications, "Si7.0-s2p2d1": the PAO name
// before the first '-', then for each angular momentum letter s p d f g the
// number of primitive radial functions (one to three digits).  With
// contraction allowed (orbital optimization), each count is written
// "primitive>contracted", e.g. "Si7.0-s3>2p2>1d1>1", with any number of
// digits.  A specification without '-', a letter other than s p d f g, a
// non-digit after a letter, a fourth digit, '>' when contraction is not
// allowed, and a contracted count above the primitive one are errors.
// Letters may repeat; the last count wins, as in the original reader.

#include <optional>
#include <string>
#include <string_view>

namespace openmx::io {

struct BasisSpec {
  std::string name;        // the characters before '-'
  int primitive[5];        // per l = s p d f g
  int contracted[5];       // per l; zero unless given after '>'
  int max_l;               // highest l with primitives, -1 when none
  bool contraction;        // whether any '>' appeared
};

// nullopt on a format error, with the original reader's message in `error`
// (when not null).
std::optional<BasisSpec> parse_basis_spec(std::string_view spec, bool allow_contraction,
                                          std::string* error = nullptr);

}  // namespace openmx::io

#endif
