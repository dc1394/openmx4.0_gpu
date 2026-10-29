#include <openmx/io/keyword_file.hpp>

#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <utility>

// Port of OpenMX's Inputtools.c (Hiori Kino, GPL).  The scan of a keyword
// (rewind, line by line, first token compared case-insensitively with equal
// length), the value tokenization, the sscanf-based conversions, the vector
// cursor arithmetic and the error texts follow the original so the
// application reads every existing input identically.

namespace openmx::io {
namespace {

bool is_space(char c) noexcept { return std::isspace(static_cast<unsigned char>(c)) != 0; }

// sscanf("%s") applied to `text`: the first whitespace-delimited token
std::string_view scan_token(std::string_view text) noexcept
{
  std::size_t begin = 0;
  while (begin < text.size() && is_space(text[begin])) ++begin;
  std::size_t end = begin;
  while (end < text.size() && !is_space(text[end])) ++end;
  return text.substr(begin, end - begin);
}

// the original nexttoken(): skips blanks and tabs only
std::size_t skip_blanks(std::string_view text, std::size_t at) noexcept
{
  if (at > text.size()) return text.size();
  while (at < text.size() && (text[at] == ' ' || text[at] == '\t')) ++at;
  return at;
}

// one fgets()-style line including its newline; false at end of file
bool read_line(std::FILE* stream, std::string& line)
{
  line.clear();
  char chunk[512];
  while (std::fgets(chunk, sizeof chunk, stream)) {
    line += chunk;
    if (!line.empty() && line.back() == '\n') return true;
  }
  return !line.empty();
}

std::string formatted(const char* format, const char* key)
{
  const int n = std::snprintf(nullptr, 0, format, key);
  std::string s(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
  std::snprintf(s.data(), s.size() + 1, format, key);
  return s;
}

std::string formatted(const char* format, const char* key, int count)
{
  const int n = std::snprintf(nullptr, 0, format, key, count);
  std::string s(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
  std::snprintf(s.data(), s.size() + 1, format, key, count);
  return s;
}

constexpr Choice logical_choices[] = {
  {"on", 1}, {"yes", 1}, {"true", 1}, {".true.", 1}, {"ok", 1},
  {"off", 0}, {"no", 0}, {"false", 0}, {".false.", 0}, {"ng", 0}};

}  // namespace

struct KeywordFile::Line {
  std::string text;       // the whole line
  std::size_t key_end;    // offset just past the keyword token
  std::string_view rest() const noexcept { return std::string_view(text).substr(key_end); }
  // the second token (sscanf("%s %s")), empty when the line has only the keyword
  std::string_view value() const noexcept { return scan_token(rest()); }
};

KeywordFile::KeywordFile(const std::string& path, std::FILE* diagnostics)
  : path_(path), stream_(std::fopen(path.c_str(), "r")), diagnostics_(diagnostics)
{
  if (!stream_) throw std::runtime_error("cannot open keyword file " + path);
}

KeywordFile::~KeywordFile() { close(); }

KeywordFile::KeywordFile(KeywordFile&& other) noexcept
  : path_(std::move(other.path_)), stream_(std::exchange(other.stream_, nullptr)),
    diagnostics_(other.diagnostics_), messages_(std::move(other.messages_))
{}

KeywordFile& KeywordFile::operator=(KeywordFile&& other) noexcept
{
  if (this != &other) {
    close();
    path_ = std::move(other.path_);
    stream_ = std::exchange(other.stream_, nullptr);
    diagnostics_ = other.diagnostics_;
    messages_ = std::move(other.messages_);
  }
  return *this;
}

int KeywordFile::close() noexcept
{
  if (!stream_) return EOF;
  const int result = std::fclose(stream_);
  stream_ = nullptr;
  return result;
}

bool KeywordFile::matches(std::string_view token, std::string_view key) noexcept
{
  if (token.size() != key.size()) return false;
  for (std::size_t i = 0; i < key.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(token[i])) != std::tolower(static_cast<unsigned char>(key[i]))) return false;
  return true;
}

bool KeywordFile::choose(std::string_view text, int& value, std::span<const Choice> choices) noexcept
{
  for (const Choice& c : choices)
    if (matches(text, c.text)) { value = c.value; return true; }
  return false;
}

void KeywordFile::fail(std::string message)
{
  if (diagnostics_) std::fputs(message.c_str(), diagnostics_);
  messages_.push_back(std::move(message));
}

bool KeywordFile::seek(std::string_view key, Line& line)
{
  if (!stream_) return false;
  std::rewind(stream_);
  while (read_line(stream_, line.text)) {
    const std::string_view text(line.text);
    const std::string_view token = scan_token(text);
    if (matches(token, key)) {
      line.key_end = static_cast<std::size_t>(token.data() - text.data()) + token.size();
      return true;
    }
  }
  return false;
}

Lookup KeywordFile::logical(std::string_view key, int& value, int fallback)
{
  value = fallback;
  Line line;
  if (!seek(key, line)) return Lookup::missing;
  const std::string_view text = line.value();
  if (!text.empty() && choose(text, value, logical_choices)) return Lookup::found;
  fail(formatted("\nERROR, key=%s value=logical\n\n", std::string(key).c_str()));
  return Lookup::invalid;
}

Lookup KeywordFile::integer(std::string_view key, int& value, int fallback)
{
  value = fallback;
  Line line;
  if (!seek(key, line)) return Lookup::missing;
  const std::string text(line.value());
  if (!text.empty() && std::sscanf(text.c_str(), "%d", &value) == 1) return Lookup::found;
  fail(formatted("\nERROR, key=%s value=int\n\n", std::string(key).c_str()));
  return Lookup::invalid;
}

Lookup KeywordFile::real(std::string_view key, double& value, double fallback)
{
  value = fallback;
  Line line;
  if (!seek(key, line)) return Lookup::missing;
  const std::string text(line.value());
  if (!text.empty() && std::sscanf(text.c_str(), "%lf", &value) == 1) return Lookup::found;
  fail(formatted("\nERROR, key=%s value=double\n\n", std::string(key).c_str()));
  return Lookup::invalid;
}

Lookup KeywordFile::string(std::string_view key, std::string& value, std::string_view fallback)
{
  value = fallback;
  Line line;
  if (!seek(key, line)) return Lookup::missing;
  const std::string_view text = line.value();
  if (!text.empty()) { value = text; return Lookup::found; }
  fail(formatted("\nERROR key=%s value=string\n\n", std::string(key).c_str()));
  return Lookup::invalid;
}

Lookup KeywordFile::choice(std::string_view key, int& value, std::span<const Choice> choices)
{
  value = choices.empty() ? 0 : choices[0].value;
  Line line;
  if (!seek(key, line)) return Lookup::missing;
  const std::string_view text = line.value();
  if (!text.empty() && choose(text, value, choices)) return Lookup::found;
  fail(formatted("\nERROR key=%s\n\n", std::string(key).c_str()));
  return Lookup::invalid;
}

// The vector forms of the original advance a cursor from one character past
// the keyword: skip blanks/tabs, read a token as sscanf("%s") would, then
// move by the token's length from where the blank skip stopped.
Lookup KeywordFile::strings(std::string_view key, std::span<std::string> values, std::span<const std::string> fallback)
{
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = i < fallback.size() ? fallback[i] : std::string();
  Line line;
  if (!seek(key, line)) return Lookup::missing;
  const std::string_view text(line.text);
  std::size_t cursor = line.key_end + 1;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::size_t at = skip_blanks(text, cursor);
    const std::string_view token = scan_token(text.substr(at));
    if (token.empty()) {
      fail(formatted("\nERROR key=%s value= %d strings\n\n", std::string(key).c_str(), static_cast<int>(values.size())));
      return Lookup::invalid;
    }
    values[i] = token;
    cursor = at + token.size();
  }
  return Lookup::found;
}

Lookup KeywordFile::reals(std::string_view key, std::span<double> values, std::span<const double> fallback)
{
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = i < fallback.size() ? fallback[i] : 0.0;
  Line line;
  if (!seek(key, line)) return Lookup::missing;
  const std::string_view text(line.text);
  std::size_t cursor = line.key_end + 1;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::size_t at = skip_blanks(text, cursor);
    const std::string token(scan_token(text.substr(at)));
    if (token.empty() || std::sscanf(token.c_str(), "%lf", &values[i]) != 1) {
      fail(formatted("\nERROR key=%s value= %d doubles\n\n", std::string(key).c_str(), static_cast<int>(values.size())));
      return Lookup::invalid;
    }
    cursor = at + token.size();
  }
  return Lookup::found;
}

Lookup KeywordFile::integers(std::string_view key, std::span<int> values, std::span<const int> fallback)
{
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = i < fallback.size() ? fallback[i] : 0;
  Line line;
  if (!seek(key, line)) return Lookup::missing;
  const std::string_view text(line.text);
  std::size_t cursor = line.key_end + 1;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::size_t at = skip_blanks(text, cursor);
    const std::string token(scan_token(text.substr(at)));
    if (token.empty() || std::sscanf(token.c_str(), "%d", &values[i]) != 1) {
      fail(formatted("\nERROR key=%s value= %d intv\n\n", std::string(key).c_str(), static_cast<int>(values.size())));
      return Lookup::invalid;
    }
    cursor = at + token.size();
  }
  return Lookup::found;
}

std::FILE* KeywordFile::find(std::string_view key)
{
  Line line;
  return seek(key, line) ? stream_ : nullptr;
}

bool KeywordFile::last(std::string_view key)
{
  if (!stream_) return false;
  std::string line;
  read_line(stream_, line);
  read_line(stream_, line);
  return matches(scan_token(line), key);
}

std::optional<std::vector<std::string>> KeywordFile::block(std::string_view name)
{
  if (!find("<" + std::string(name))) return std::nullopt;
  const std::string closing = std::string(name) + ">";
  std::vector<std::string> lines;
  std::string line;
  while (read_line(stream_, line)) {
    if (matches(scan_token(line), closing)) return lines;
    if (!line.empty() && line.back() == '\n') line.pop_back();
    if (!line.empty() && line.back() == '\r') line.pop_back();
    lines.push_back(line);
  }
  return std::nullopt;
}

KeywordFiles::KeywordFiles(int capacity, std::FILE* diagnostics)
  : capacity_(capacity < 1 ? 1 : capacity), diagnostics_(diagnostics)
{}

int KeywordFiles::total_errors() const noexcept
{
  int total = closed_errors_;
  for (const KeywordFile& f : files_) total += f.error_count();
  return total;
}

bool KeywordFiles::open(const std::string& path)
{
  if (depth() >= capacity_) {
    std::string message = "input_open: can not open a further file. nestlevel=" + std::to_string(depth()) + "\n";
    if (diagnostics_) std::fputs(message.c_str(), diagnostics_);
    messages_.push_back(std::move(message));
    return false;
  }
  try {
    files_.emplace_back(path, diagnostics_);
  } catch (const std::exception&) {
    base_ = total_errors();
    std::string message = "input_open: can not open " + path + "\n";
    if (diagnostics_) std::fputs(message.c_str(), diagnostics_);
    messages_.push_back(std::move(message));
    return false;
  }
  base_ = total_errors();
  return true;
}

int KeywordFiles::close()
{
  if (files_.empty()) return EOF;
  closed_errors_ += files_.back().error_count();
  const int result = files_.back().close();
  files_.pop_back();
  return result;
}

int KeywordFiles::error_count() const noexcept { return total_errors() - base_; }

}  // namespace openmx::io
