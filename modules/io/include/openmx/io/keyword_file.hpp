#ifndef OPENMX_IO_KEYWORD_FILE_HPP
#define OPENMX_IO_KEYWORD_FILE_HPP

// Reader of OpenMX keyword files: every line whose first whitespace-separated
// token equals a keyword (case-insensitively, same length) carries that
// keyword's value(s) in the following tokens; blocks are enclosed in a line
// "<Keyword" and a line "Keyword>".  A KeywordFile owns the open stream and
// records its own error count and messages; KeywordFiles is the nesting model
// of the original application (a stack of files, the top one answering, one
// error counter reset by each open).  No global state, no process exit, no
// MPI.  The lookup rules, the accepted spellings, the sscanf-based number
// parsing and the error texts are those of OpenMX's Inputtools.c
// (Hiori Kino), including its quirks, so the application reads its inputs
// unchanged; see README.md for the few deliberate differences.

#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace openmx::io {

// 1 / 0 / -1 of the original API.
enum class Lookup { found = 1, missing = 0, invalid = -1 };

// one accepted spelling of a keyword's value and the integer it stands for
struct Choice {
  const char* text;
  int value;
};

class KeywordFile {
public:
  // Opens `path` for reading or throws std::runtime_error.  `diagnostics`,
  // when not null, receives every error message as it is recorded.
  explicit KeywordFile(const std::string& path, std::FILE* diagnostics = nullptr);
  ~KeywordFile();
  KeywordFile(KeywordFile&& other) noexcept;
  KeywordFile& operator=(KeywordFile&& other) noexcept;
  KeywordFile(const KeywordFile&) = delete;
  KeywordFile& operator=(const KeywordFile&) = delete;

  const std::string& path() const noexcept { return path_; }
  // The owned stream (never null for a live object; null after a move).
  std::FILE* stream() const noexcept { return stream_; }

  // Scalar keywords: the second token of the keyword's line.  The value is
  // set to `fallback` first; `missing` leaves it so, `invalid` (a line
  // without a value, or a value of the wrong form) records an error.
  Lookup logical(std::string_view key, int& value, int fallback);  // on yes true .true. ok / off no false .false. ng
  Lookup integer(std::string_view key, int& value, int fallback);
  Lookup real(std::string_view key, double& value, double fallback);
  Lookup string(std::string_view key, std::string& value, std::string_view fallback);
  // The value must be one of `choices` (case-insensitively); the fallback is
  // choices[0].value.  `choices` must not be empty.
  Lookup choice(std::string_view key, int& value, std::span<const Choice> choices);

  // Vector keywords: `values.size()` tokens after the keyword.  Every element
  // takes its fallback first (fallback.size() >= values.size()); elements
  // parsed before a failure keep their parsed value.
  Lookup strings(std::string_view key, std::span<std::string> values, std::span<const std::string> fallback);
  Lookup reals(std::string_view key, std::span<double> values, std::span<const double> fallback);
  Lookup integers(std::string_view key, std::span<int> values, std::span<const int> fallback);

  // Blocks.  find() positions the stream right after the line whose first
  // token is `key` (usually "<Name") and returns it for line-wise reading, or
  // returns null when absent.  last() then consumes the rest of the current
  // line and one more line, and tells whether that line's first token is
  // `key` (usually "Name>").  block() returns the lines strictly between the
  // tags "<name" and "name>", or nullopt when either is absent.
  std::FILE* find(std::string_view key);
  bool last(std::string_view key);
  std::optional<std::vector<std::string>> block(std::string_view name);

  // Closes the stream early (the destructor does it otherwise); fclose's result, EOF when already closed.
  int close() noexcept;

  int error_count() const noexcept { return static_cast<int>(messages_.size()); }
  const std::vector<std::string>& messages() const noexcept { return messages_; }

  // The matching rule of keywords and of choices: equal length and
  // case-insensitively equal characters.
  static bool matches(std::string_view token, std::string_view key) noexcept;
  // Resolves `text` among `choices` by the matching rule; false when none fits.
  static bool choose(std::string_view text, int& value, std::span<const Choice> choices) noexcept;

private:
  struct Line;
  bool seek(std::string_view key, Line& line);
  void fail(std::string message);

  std::string path_;
  std::FILE* stream_ = nullptr;
  std::FILE* diagnostics_ = nullptr;
  std::vector<std::string> messages_;
};

// The nesting model of the original Inputtools: open() pushes a file, close()
// pops it, lookups go to the top file, and error_count() counts the errors
// recorded (in any file) since the last successful open.
class KeywordFiles {
public:
  explicit KeywordFiles(int capacity = 14, std::FILE* diagnostics = nullptr);
  ~KeywordFiles() = default;
  KeywordFiles(KeywordFiles&&) noexcept = default;
  KeywordFiles& operator=(KeywordFiles&&) noexcept = default;
  KeywordFiles(const KeywordFiles&) = delete;
  KeywordFiles& operator=(const KeywordFiles&) = delete;

  // false (with a message) when the stack is full or the file cannot be read.
  bool open(const std::string& path);
  // fclose's result for the top file, EOF when nothing is open.
  int close();
  int depth() const noexcept { return static_cast<int>(files_.size()); }
  int capacity() const noexcept { return capacity_; }
  // The top file, or null when nothing is open.
  KeywordFile* current() noexcept { return files_.empty() ? nullptr : &files_.back(); }
  const KeywordFile* current() const noexcept { return files_.empty() ? nullptr : &files_.back(); }
  int error_count() const noexcept;
  // Messages of the stack itself (open failures), not of the files.
  const std::vector<std::string>& messages() const noexcept { return messages_; }

private:
  int total_errors() const noexcept;

  int capacity_;
  std::FILE* diagnostics_;
  std::vector<KeywordFile> files_;
  std::vector<std::string> messages_;
  int closed_errors_ = 0;  // errors of files already closed
  int base_ = 0;           // total errors when the last file was opened
};

}  // namespace openmx::io

#endif
