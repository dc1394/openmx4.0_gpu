// The basis-specification grammar: the counts of the original reader for
// valid texts, its two diagnostics for malformed ones, and the C interface.
#include <openmx/io/basis_spec.h>
#include <openmx/io/basis_spec.hpp>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
using openmx::io::BasisSpec;
using openmx::io::parse_basis_spec;

void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

BasisSpec parsed(const char* text, bool contraction = false) {
  std::string error;
  const auto spec = parse_basis_spec(text, contraction, &error);
  if (!spec) throw std::runtime_error(std::string("unexpected error for ") + text + ": " + error);
  return *spec;
}

void same_counts(const BasisSpec& s, int s0, int p0, int d0, int f0, int g0, int max_l) {
  require(s.primitive[0] == s0 && s.primitive[1] == p0 && s.primitive[2] == d0 && s.primitive[3] == f0 && s.primitive[4] == g0, "primitive counts");
  require(s.max_l == max_l, "max_l");
}

void rejected(const char* text, bool contraction, const char* expected) {
  std::string error;
  require(!parse_basis_spec(text, contraction, &error), text);
  require(error.find(expected) != std::string::npos, "diagnostic text");
}
}  // namespace

int main() {
  try {
    BasisSpec s = parsed("Si7.0-s2p2d1");
    require(s.name == "Si7.0" && !s.contraction, "name");
    same_counts(s, 2, 2, 1, 0, 0, 2);
    require(s.contracted[0] == 0 && s.contracted[2] == 0, "no contraction given");
    same_counts(parsed("H5.0-s2p1"), 2, 1, 0, 0, 0, 1);
    same_counts(parsed("Fe8.0S-s3p2d2f1"), 3, 2, 2, 1, 0, 3);
    require(parsed("Fe8.0S-s3p2d2f1").name == "Fe8.0S", "name with a suffix");
    same_counts(parsed("X-s123p45d6g7"), 123, 45, 6, 0, 7, 4);
    same_counts(parsed("Kr10.0-s1"), 1, 0, 0, 0, 0, 0);
    same_counts(parsed("Y-s2s3"), 3, 0, 0, 0, 0, 0);          // a repeated letter: the last count wins
    same_counts(parsed("Z-"), 0, 0, 0, 0, 0, -1);             // no counts at all
    same_counts(parsed("Z-sp"), 0, 0, 0, 0, 0, -1);           // letters without digits
    require(parsed("a.b_c-s1").name == "a.b_c", "the name is everything before the dash");
    // contraction: primitive>contracted per letter
    s = parsed("Si7.0-s3>2p2>1d1>1", true);
    require(s.contraction, "contraction flag");
    same_counts(s, 3, 2, 1, 0, 0, 2);
    require(s.contracted[0] == 2 && s.contracted[1] == 1 && s.contracted[2] == 1 && s.contracted[3] == 0, "contracted counts");
    s = parsed("C6.0-s12>10p3>3", true);
    require(s.primitive[0] == 12 && s.contracted[0] == 10 && s.primitive[1] == 3 && s.contracted[1] == 3, "multi-digit contraction");
    // the original reader's diagnostics
    rejected("Si7.0", false, "Format error");                 // no '-'
    rejected("Si7.0-x2", false, "Format error");              // not an angular momentum letter
    rejected("Si7.0-2", false, "Format error");               // digits before any letter
    rejected("Si7.0-s2x", false, "Format error");             // a non-digit count
    rejected("Si7.0-s1234", false, "Format error");           // a fourth digit without contraction
    rejected("Si7.0-s2>1", false, "Format error");            // '>' without contraction
    rejected("Si7.0-S2", false, "Format error");              // letters are lower case
    rejected("Si7.0-s1>2", true, "contracted orbitals are larger");
    rejected("Si7.0-s2p1", true, "contracted orbitals are larger");   // contraction without '>' gives zero primitives
    require(parsed("Si7.0-s2>", true).primitive[0] == 2 && parsed("Si7.0-s2>", true).contracted[0] == 0, "'>' without a contracted count");
    require(parse_basis_spec("", false) == std::nullopt, "empty text");
    // C interface
    char name[16], message[200];
    OpenMXBasisSpec c;
    require(openmx_io_parse_basis_spec("Si7.0-s2p2d1", 0, name, sizeof name, &c, message, sizeof message) == 1, "C parse");
    require(std::strcmp(name, "Si7.0") == 0 && c.primitive[1] == 2 && c.max_l == 2 && !c.contraction && message[0] == '\0', "C result");
    require(openmx_io_parse_basis_spec("Si7.0-s2>1", 0, name, sizeof name, &c, message, sizeof message) == 0, "C format error");
    require(std::strcmp(message, "Format error in Definition of Atomic Species\n") == 0, "C message");
    require(openmx_io_parse_basis_spec("VeryLongBasisName-s1", 0, name, 8, &c, message, sizeof message) == 0, "name too long");
    require(openmx_io_parse_basis_spec(nullptr, 0, name, sizeof name, &c, nullptr, 0) == 0, "null text");
    require(openmx_io_parse_basis_spec("H5.0-s2p1", 0, name, sizeof name, nullptr, nullptr, 0) == 0, "null output");
    std::printf("PASS io basis spec\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
