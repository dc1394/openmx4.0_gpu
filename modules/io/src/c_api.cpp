#include <openmx/io/keyword_file.h>
#include <openmx/io/keyword_file.hpp>

#include <cstring>
#include <new>
#include <string>
#include <vector>

struct OpenMXKeywordFiles {
  openmx::io::KeywordFiles files;
};

openmx::io::KeywordFile* openmx_io_keyword_files_current(OpenMXKeywordFiles* files) noexcept
{
  return files ? files->files.current() : nullptr;
}

namespace {

int code(openmx::io::Lookup lookup) noexcept { return static_cast<int>(lookup); }

openmx::io::KeywordFile* top(OpenMXKeywordFiles* files) noexcept
{
  return files ? files->files.current() : nullptr;
}

std::vector<openmx::io::Choice> choices_of(int count, const char* const* texts, const int* values)
{
  std::vector<openmx::io::Choice> choices;
  for (int i = 0; i < count; ++i) choices.push_back({texts[i], values[i]});
  return choices;
}

// copies text into value[capacity]; false when it does not fit
bool store(const std::string& text, char* value, size_t capacity) noexcept
{
  if (!value || capacity == 0 || text.size() >= capacity) return false;
  std::memcpy(value, text.c_str(), text.size() + 1);
  return true;
}

}  // namespace

extern "C" OpenMXKeywordFiles* openmx_io_keyword_files_create(int capacity, FILE* diagnostics)
{
  try {
    return new OpenMXKeywordFiles{openmx::io::KeywordFiles(capacity, diagnostics)};
  } catch (...) {
    return nullptr;
  }
}

extern "C" void openmx_io_keyword_files_destroy(OpenMXKeywordFiles* files) { delete files; }

extern "C" int openmx_io_open(OpenMXKeywordFiles* files, const char* path)
{
  if (!files || !path) return 0;
  try {
    return files->files.open(path) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

extern "C" int openmx_io_close(OpenMXKeywordFiles* files) { return files ? files->files.close() : EOF; }
extern "C" int openmx_io_depth(const OpenMXKeywordFiles* files) { return files ? files->files.depth() : 0; }
extern "C" int openmx_io_error_count(const OpenMXKeywordFiles* files) { return files ? files->files.error_count() : 0; }

extern "C" int openmx_io_logical(OpenMXKeywordFiles* files, const char* key, int* value, int fallback)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key || !value) return -1;
  try { return code(file->logical(key, *value, fallback)); } catch (...) { return -1; }
}

extern "C" int openmx_io_int(OpenMXKeywordFiles* files, const char* key, int* value, int fallback)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key || !value) return -1;
  try { return code(file->integer(key, *value, fallback)); } catch (...) { return -1; }
}

extern "C" int openmx_io_double(OpenMXKeywordFiles* files, const char* key, double* value, double fallback)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key || !value) return -1;
  try { return code(file->real(key, *value, fallback)); } catch (...) { return -1; }
}

extern "C" int openmx_io_string(OpenMXKeywordFiles* files, const char* key, char* value, size_t capacity, const char* fallback)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key || !value || !fallback || capacity == 0) return -1;
  try {
    std::string text;
    const openmx::io::Lookup lookup = file->string(key, text, fallback);
    if (lookup == openmx::io::Lookup::found && !store(text, value, capacity)) {
      store(std::string(fallback).substr(0, capacity - 1), value, capacity);
      return -1;
    }
    if (lookup != openmx::io::Lookup::found) store(std::string(fallback).substr(0, capacity - 1), value, capacity);
    return code(lookup);
  } catch (...) {
    return -1;
  }
}

extern "C" int openmx_io_string2int(OpenMXKeywordFiles* files, const char* key, int* value,
                                    int count, const char* const* texts, const int* values)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key || !value || count < 1 || !texts || !values) return -1;
  try { return code(file->choice(key, *value, choices_of(count, texts, values))); } catch (...) { return -1; }
}

extern "C" int openmx_io_stringv(OpenMXKeywordFiles* files, const char* key, int count,
                                 char* const* values, size_t capacity, const char* const* fallback)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key || count < 0 || (count > 0 && (!values || !fallback)) || capacity == 0) return -1;
  try {
    std::vector<std::string> texts(static_cast<std::size_t>(count)), defaults;
    for (int i = 0; i < count; ++i) defaults.emplace_back(fallback[i] ? fallback[i] : "");
    const openmx::io::Lookup lookup = file->strings(key, texts, defaults);
    int result = code(lookup);
    for (int i = 0; i < count; ++i)
      if (!store(texts[static_cast<std::size_t>(i)], values[i], capacity)) {
        store(defaults[static_cast<std::size_t>(i)].substr(0, capacity - 1), values[i], capacity);
        result = -1;
      }
    return result;
  } catch (...) {
    return -1;
  }
}

extern "C" int openmx_io_doublev(OpenMXKeywordFiles* files, const char* key, int count,
                                 double* values, const double* fallback)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key || count < 0 || (count > 0 && (!values || !fallback))) return -1;
  try {
    return code(file->reals(key, std::span<double>(values, static_cast<std::size_t>(count)),
                            std::span<const double>(fallback, static_cast<std::size_t>(count))));
  } catch (...) {
    return -1;
  }
}

extern "C" int openmx_io_intv(OpenMXKeywordFiles* files, const char* key, int count,
                              int* values, const int* fallback)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key || count < 0 || (count > 0 && (!values || !fallback))) return -1;
  try {
    return code(file->integers(key, std::span<int>(values, static_cast<std::size_t>(count)),
                               std::span<const int>(fallback, static_cast<std::size_t>(count))));
  } catch (...) {
    return -1;
  }
}

extern "C" FILE* openmx_io_find(OpenMXKeywordFiles* files, const char* key)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key) return nullptr;
  try { return file->find(key); } catch (...) { return nullptr; }
}

extern "C" int openmx_io_last(OpenMXKeywordFiles* files, const char* key)
{
  openmx::io::KeywordFile* file = top(files);
  if (!file || !key) return 0;
  try { return file->last(key) ? 1 : 0; } catch (...) { return 0; }
}

extern "C" int openmx_io_cmpstring(const char* text, int* value, int count, const char* const* texts, const int* values)
{
  if (!text || !value || count < 1 || !texts || !values) return 0;
  try {
    return openmx::io::KeywordFile::choose(text, *value, choices_of(count, texts, values)) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}
