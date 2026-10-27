#pragma once

// Private PAO/VPS text reader. Owns its bytes; token/line views never escape.
#include <charconv>
#include <cmath>
#include <istream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace openmx::basis::detail {
inline bool space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v'; }
inline bool equal(std::string_view a, std::string_view b)
{
  if (a.size() != b.size()) return false;
  auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c-'A'+'a') : c; };
  for (std::size_t i = 0; i < a.size(); ++i) if (lower(a[i]) != lower(b[i])) return false;
  return true;
}

class Tokens {
 public:
  Tokens(std::string_view text, std::string section, std::size_t line)
    : text_(text), section_(std::move(section)), line_(line) {}
  std::string_view next() {
    while (!text_.empty()) {
      if (text_.front() == '#') {
        const auto end = text_.find('\n');
        if (end == std::string_view::npos) { text_ = {}; break; }
        text_.remove_prefix(end);
      } else if (!space(text_.front())) break;
      if (text_.front() == '\n') ++line_;
      text_.remove_prefix(1);
    }
    const auto end = text_.find_first_of(" \t\r\n\f\v#");
    const auto token = text_.substr(0,end);
    text_.remove_prefix(token.size());
    return token;
  }
  template<class Number> Number number() {
    auto token = next();
    if (!token.empty() && token.front() == '+') {
      token.remove_prefix(1);
      if (!token.empty() && (token.front() == '+' || token.front() == '-')) fail("invalid numeric sign");
    }
    Number value{};
    if (token.empty()) fail("missing numeric value");
    const auto parsed = std::from_chars(token.data(),token.data()+token.size(),value);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data()+token.size() || !std::isfinite(value))
      fail("invalid numeric value '"+std::string(token)+"'");
    return value;
  }
  void end() { if (!next().empty()) fail("extra values in table"); }
  [[noreturn]] void fail(const std::string& reason) const {
    throw std::runtime_error(section_+" at line "+std::to_string(line_)+": "+reason);
  }
 private:
  std::string_view text_;
  std::string section_;
  std::size_t line_;
};

class Document {
 public:
  explicit Document(std::istream& input, std::string format)
    : format_(std::move(format)), text_(std::istreambuf_iterator<char>(input),{}) {
    if (input.bad()) throw std::runtime_error(format_+" input read failed");
    if (text_.find('\0') != std::string::npos) throw std::runtime_error(format_+" input contains a NUL byte");
    std::size_t begin = 0;
    while (begin < text_.size()) {
      const auto end = text_.find('\n',begin);
      lines_.push_back({std::string_view(text_).substr(begin,end == std::string::npos ? end : end-begin),begin});
      if (end == std::string::npos) break;
      begin = end+1;
    }
  }
  Document(const Document&) = delete;
  Document& operator=(const Document&) = delete;
  std::size_t size() const noexcept { return text_.size(); }
  std::optional<std::size_t> find(std::string_view key, std::size_t first = 0) const {
    for (std::size_t i = first; i < lines_.size(); ++i) {
      Tokens tokens(lines_[i].text,"",i+1);
      if (equal(tokens.next(),key)) return i;
    }
    return std::nullopt;
  }
  template<class Number> Number scalar(std::string_view key, std::optional<Number> fallback = {}) const {
    const auto index = find(key);
    if (!index) {
      if (fallback) return *fallback;
      throw std::runtime_error(format_+" missing scalar "+std::string(key));
    }
    Tokens tokens(lines_[*index].text,format_+" "+std::string(key),*index+1);
    tokens.next();
    return tokens.number<Number>();
  }
  bool logical(std::string_view key, bool fallback = false) const {
    const auto index = find(key);
    if (!index) return fallback;
    Tokens tokens(lines_[*index].text,format_+" "+std::string(key),*index+1);
    tokens.next();
    const auto value = tokens.next();
    for (const auto truth : {"on","yes","true",".true.","ok"}) if (equal(value,truth)) return true;
    for (const auto falsity : {"off","no","false",".false.","ng"}) if (equal(value,falsity)) return false;
    tokens.fail("invalid logical value");
  }
  Tokens section(const std::string& name) const {
    const auto begin = find("<"+name);
    if (!begin) throw std::runtime_error(format_+" missing section "+name);
    const auto end = find(name+">",*begin+1);
    if (!end) throw std::runtime_error(format_+" missing closing tag for "+name);
    const auto offset = lines_[*begin+1].offset;
    return {std::string_view(text_).substr(offset,lines_[*end].offset-offset),format_+" "+name,*begin+2};
  }
  Tokens after(std::size_t line, const std::string& name) const {
    const auto offset = line+1 < lines_.size() ? lines_[line+1].offset : text_.size();
    return {std::string_view(text_).substr(offset),format_+" "+name,line+2};
  }
 private:
  struct Line { std::string_view text; std::size_t offset; };
  std::string format_, text_;
  std::vector<Line> lines_;
};

inline std::size_t checked_product(std::size_t a, std::size_t b)
{
  if (b && a > std::vector<double>().max_size()/b)
    throw std::length_error("radial table dimensions overflow");
  return a*b;
}

inline void read_mesh_point(Tokens& input, std::size_t i,
                     std::vector<double>& logarithms, std::vector<double>& radii)
{
  logarithms[i] = input.number<double>();
  radii[i] = input.number<double>();
  if (radii[i] <= 0 || (i && (radii[i] <= radii[i-1] || logarithms[i] <= logarithms[i-1])))
    input.fail("mesh must be positive and strictly increasing");
}

} // namespace openmx::basis::detail
