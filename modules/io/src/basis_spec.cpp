#include <openmx/io/basis_spec.h>
#include <openmx/io/basis_spec.hpp>

#include <cstdio>
#include <cstring>

namespace openmx::io {

namespace {

int angular_index(char c) noexcept
{
  switch (c) {
  case 's': return 0;
  case 'p': return 1;
  case 'd': return 2;
  case 'f': return 3;
  case 'g': return 4;
  default: return -1;
  }
}

std::optional<BasisSpec> fail(std::string* error, std::string message)
{
  if (error) *error = std::move(message);
  return std::nullopt;
}

}  // namespace

std::optional<BasisSpec> parse_basis_spec(std::string_view spec, bool allow_contraction, std::string* error)
{
  // The original reader's two diagnostics.
  static constexpr const char* format_error = "Format error in Definition of Atomic Species\n";
  const std::size_t dash = spec.find('-');
  if (dash == std::string_view::npos) return fail(error, format_error);

  BasisSpec result;
  result.name = std::string(spec.substr(0, dash));
  for (int l = 0; l < 5; ++l) { result.primitive[l] = 0; result.contracted[l] = 0; }
  result.max_l = -1;
  result.contraction = false;

  int l = -1;           // the current angular momentum
  int digits = 0;       // digits read since the letter or the last '>'
  int value = 0;        // their value
  bool after_arrow = false;
  auto finish_pending = [&]() {
    // digits that follow a letter without '>' are the contracted count of
    // the original reader (which then rejects them against zero primitives)
    if (digits && allow_contraction && !after_arrow) result.contracted[l] = value;
  };
  for (std::size_t i = dash + 1; i < spec.size(); ++i) {
    const char c = spec[i];
    const int next = angular_index(c);
    if (next >= 0) {
      if (l >= 0) finish_pending();
      l = next;
      digits = 0;
      value = 0;
      after_arrow = false;
      continue;
    }
    if (l < 0) return fail(error, format_error);
    if (c == '>') {
      if (!allow_contraction) return fail(error, format_error);
      result.primitive[l] = value;
      result.contraction = true;
      after_arrow = true;
      digits = 0;
      value = 0;
      continue;
    }
    if (c < '0' || c > '9') return fail(error, format_error);
    if (!allow_contraction && digits == 3) return fail(error, format_error);
    value = 10 * value + (c - '0');
    ++digits;
    if (!allow_contraction) result.primitive[l] = value;
    else if (after_arrow) result.contracted[l] = value;
  }
  if (l >= 0) finish_pending();

  for (int m = 0; m < 5; ++m) {
    if (result.primitive[m] != 0) result.max_l = m;
    if (result.primitive[m] < result.contracted[m]) {
      char text[160];
      std::snprintf(text, sizeof text,
                    "# of contracted orbitals are larger than # of primitive oribitals\nPrimitive=%3d Contracted=%3d\n",
                    result.primitive[m], result.contracted[m]);
      return fail(error, text);
    }
  }
  return result;
}

}  // namespace openmx::io

extern "C" int openmx_io_parse_basis_spec(const char* text, int allow_contraction,
                                          char* name, size_t capacity, OpenMXBasisSpec* spec,
                                          char* message, size_t message_capacity)
{
  if (message && message_capacity) message[0] = '\0';
  if (!text || !name || capacity == 0 || !spec) return 0;
  try {
    std::string error;
    const std::optional<openmx::io::BasisSpec> parsed =
        openmx::io::parse_basis_spec(text, allow_contraction != 0, &error);
    if (!parsed) {
      if (message && message_capacity) std::snprintf(message, message_capacity, "%s", error.c_str());
      return 0;
    }
    if (parsed->name.size() >= capacity) {
      if (message && message_capacity) std::snprintf(message, message_capacity, "basis name longer than %zu characters\n", capacity - 1);
      return 0;
    }
    std::memcpy(name, parsed->name.c_str(), parsed->name.size() + 1);
    for (int l = 0; l < 5; ++l) { spec->primitive[l] = parsed->primitive[l]; spec->contracted[l] = parsed->contracted[l]; }
    spec->max_l = parsed->max_l;
    spec->contraction = parsed->contraction ? 1 : 0;
    return 1;
  } catch (...) {
    return 0;
  }
}
