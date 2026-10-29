#include "../src/text_document.hpp"

#include <array>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition) { if (!condition) throw std::runtime_error("numeric token contract failed"); }
template<class Number, bool Fast> Number read(std::string_view token)
{
  openmx::basis::detail::Tokens input(token,"numeric fixture",7);
  const auto result = input.number<Number,Fast>();
  input.end();
  return result;
}
template<class Number, bool Fast> void rejects(std::string_view token)
{
  try { (void)read<Number,Fast>(token); }
  catch (const std::runtime_error& error) {
    require(std::string_view(error.what()).find("numeric fixture at line 7:") == 0);
    return;
  }
  throw std::runtime_error("invalid numeric token accepted: "+std::string(token));
}
template<bool Fast> void numbers()
{
  struct Case { std::string_view text; double value; };
  const std::array cases{
    Case{"0",0.},Case{"-0.0e+12",-0.},Case{"+0",0.},Case{"+.5",.5},Case{"2.",2.},
    Case{"-3.125E-2",-.03125},Case{"0.1",.1},Case{"1.2345678901234567e-12",1.2345678901234567e-12},
    Case{"1.7976931348623157e308",std::numeric_limits<double>::max()},
    Case{"2.2250738585072014e-308",std::numeric_limits<double>::min()},
    Case{"2.2250738585072012e-308",std::numeric_limits<double>::min()},
    Case{"2.2250738585072011e-308",std::nextafter(std::numeric_limits<double>::min(),0.)},
    Case{"4.9406564584124654e-324",std::numeric_limits<double>::denorm_min()},
    Case{"2.4703282292062328e-324",std::numeric_limits<double>::denorm_min()},
    Case{"0e-9999",0.},Case{"-0e9999",-0.}};
  for (const auto& test : cases) {
    const auto value = read<double,Fast>(test.text);
    require(value == test.value && std::signbit(value) == std::signbit(test.value));
  }
  for (const auto text : {"", "+", "-", "++1", "+-1", "--1", "1e", "1e+", "1.0x", "1,5",
                          "0x1p2", "NaN", "inf", "-infinity", "1e309", "-1e309", "1e-9999", "-1e-9999",
                          "2.4703282292062327e-324", "1e-308.0", "1e-308e1", "-1e-308-1"})
    rejects<double,Fast>(text);
  require((read<int,Fast>("+2147483647") == std::numeric_limits<int>::max()));
  require((read<int,Fast>("-2147483648") == std::numeric_limits<int>::min()));
  for (const auto text : {"2147483648", "-2147483649", "1.0", "1e2", "0x10"}) rejects<int,Fast>(text);
}
struct Comma : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};
struct RestoreLocale {
  std::locale saved = std::locale();
  ~RestoreLocale() { std::locale::global(saved); }
};
}

int main()
{
  numbers<true>();
  // Exercise the libc++ compatibility path even on a library with from_chars.
  numbers<false>();
  RestoreLocale restore;
  std::locale::global(std::locale(std::locale::classic(),new Comma));
  numbers<true>();
  numbers<false>();
}
