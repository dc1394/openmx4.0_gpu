// Tests of the io package: behavioural agreement with the frozen Inputtools.c
// of OpenMX 4.0 (return codes, values, error counts and the diagnostics text),
// the C++ ownership semantics, and the boundary behaviour of the C interface.
#include <openmx/io/keyword_file.h>
#include <openmx/io/keyword_file.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "Inputtools.h"   // the frozen original: one process-wide file stack
}

namespace {
using openmx::io::KeywordFile;
using openmx::io::KeywordFiles;
using openmx::io::Lookup;

void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
void same(const std::string& what, long legacy, long library) {
  if (legacy != library) throw std::runtime_error(what + ": legacy " + std::to_string(legacy) + " vs library " + std::to_string(library));
}
void same_bits(const std::string& what, double legacy, double library) {
  if (std::memcmp(&legacy, &library, sizeof legacy) != 0) throw std::runtime_error(what + ": legacy " + std::to_string(legacy) + " vs library " + std::to_string(library));
}
void same(const std::string& what, const std::string& legacy, const std::string& library) {
  if (legacy != library) throw std::runtime_error(what + ": legacy '" + legacy + "' vs library '" + library + "'");
}

struct TempDir {
  std::filesystem::path path;
  TempDir() {
    path = std::filesystem::temp_directory_path() / ("openmx-io-test-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(path);
  }
  ~TempDir() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
  std::string write(const std::string& name, const std::string& content) const {
    const std::filesystem::path file = path / name;
    std::ofstream(file, std::ios::binary) << content;
    return file.string();
  }
};

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

const char* const basic_text =
  "System.Name   test1   # comment\n"
  "scf.maxIter   40\n"
  "scf.energycutoff   150.0\n"
  "scf.SpinPolarization   on\n"
  "scf.XcType   GGA-PBE\n"
  "Dos.Erange  -10.0 10.0\n"
  "scf.Kgrid   4 4 4\n"
  "strings.key  alpha beta gamma\n"
  "dup.key 1\n"
  "dup.key 2\n"
  "<Definition.of.Atomic.Species\n"
  " H   H5.0-s2p1   H_PBE19\n"
  " O   O5.0-s2p2d1 O_PBE19\n"
  "Definition.of.Atomic.Species>\n"
  "Atoms.Number  3\n"
  "<Atoms.SpeciesAndCoordinates\n"
  " 1  O  0.000 0.000 0.000  3.0 3.0\n"
  " 2  H  0.757 0.586 0.000  0.5 0.5\n"
  " 3  H -0.757 0.586 0.000  0.5 0.5\n"
  "Atoms.SpeciesAndCoordinates>\n"
  "<Broken.Block\n"
  " 1 2 3\n"
  "Not.The.Closing.Tag\n"
  "Tabbed.Key\t7\n"
  "Mixed.CASE.key  value\n"
  "empty.value\n"
  "bad.int  abc\n"
  "bad.double  x1.5\n"
  "bad.logical  maybe\n"
  "partial.vector  1.0 2.0\n"
  "int.with.suffix 12abc\n"
  "hex.int 0x1A\n"
  "sci.double 1.5e-3\n"
  "logical.false .FALSE.\n";

// Every lookup of the original API against the library, on the file open on
// both sides.  The original input_double returns 0 for an invalid value where
// every other lookup returns -1; the library returns -1 uniformly.
struct Pair {
  OpenMXKeywordFiles* lib;
  void integer(const char* key, int fallback) const {
    int a = 0, b = 0;
    same(std::string("int ") + key, input_int(key, &a, fallback), openmx_io_int(lib, key, &b, fallback));
    same(std::string("int value ") + key, a, b);
  }
  void real(const char* key, double fallback) const {
    double a = 0, b = 0;
    const int lib_code = openmx_io_double(lib, key, &b, fallback);
    same(std::string("double ") + key, input_double(key, &a, fallback), lib_code < 0 ? 0 : lib_code);
    same_bits(std::string("double value ") + key, a, b);
  }
  void logical(const char* key, int fallback) const {
    int a = 0, b = 0;
    same(std::string("logical ") + key, input_logical(key, &a, fallback), openmx_io_logical(lib, key, &b, fallback));
    same(std::string("logical value ") + key, a, b);
  }
  void string(const char* key, const char* fallback) const {
    char a[256], b[256];
    same(std::string("string ") + key, input_string(key, a, fallback), openmx_io_string(lib, key, b, sizeof b, fallback));
    same(std::string("string value ") + key, std::string(a), std::string(b));
  }
  void choice(const char* key) const {
    static const char* names[3] = {"LDA", "LSDA-CA", "GGA-PBE"};
    static int values[3] = {1, 2, 4};
    int a = 0, b = 0;
    same(std::string("string2int ") + key, input_string2int(key, &a, 3, const_cast<char**>(names), values),
         openmx_io_string2int(lib, key, &b, 3, names, values));
    same(std::string("string2int value ") + key, a, b);
  }
  void strings(const char* key, int count) const {
    std::vector<std::string> storage_a(count, std::string(64, '\0')), storage_b(count, std::string(64, '\0'));
    std::vector<char*> a, b;
    std::vector<const char*> defaults;
    for (int i = 0; i < count; ++i) { a.push_back(storage_a[i].data()); b.push_back(storage_b[i].data()); defaults.push_back("default"); }
    same(std::string("stringv ") + key, input_stringv(key, count, a.data(), const_cast<char**>(defaults.data())),
         openmx_io_stringv(lib, key, count, b.data(), 64, defaults.data()));
    for (int i = 0; i < count; ++i) same(std::string("stringv value ") + key, std::string(a[i]), std::string(b[i]));
  }
  void reals(const char* key, int count) const {
    std::vector<double> a(count), b(count), defaults(count, -1.0);
    same(std::string("doublev ") + key, input_doublev(key, count, a.data(), defaults.data()),
         openmx_io_doublev(lib, key, count, b.data(), defaults.data()));
    for (int i = 0; i < count; ++i) same_bits(std::string("doublev value ") + key, a[i], b[i]);
  }
  void integers(const char* key, int count) const {
    std::vector<int> a(count), b(count), defaults(count, -1);
    same(std::string("intv ") + key, input_intv(key, count, a.data(), defaults.data()),
         openmx_io_intv(lib, key, count, b.data(), defaults.data()));
    for (int i = 0; i < count; ++i) same(std::string("intv value ") + key, a[i], b[i]);
  }
  void errors(const char* when) const { same(std::string("error count ") + when, input_errorCount(), openmx_io_error_count(lib)); }
  void open(const std::string& path) const { same("open " + path, input_open(path.c_str()), openmx_io_open(lib, path.c_str())); }
  void close() const { same("close", input_close(), openmx_io_close(lib)); }
  // the two block idioms of Input_std.c: fscanf per line, and fgets per line followed by ungetc('\n')
  void block_fscanf(const char* opening, const char* closing, int lines) const {
    std::FILE* a = input_find(opening);
    std::FILE* b = openmx_io_find(lib, opening);
    require((a != nullptr) == (b != nullptr), std::string("find ") + opening);
    if (!a) return;
    for (int i = 0; i < lines; ++i) {
      char x[64], y[64], z[64], p[64], q[64], r[64];
      same(std::string("fscanf ") + opening, std::fscanf(a, "%63s %63s %63s", x, y, z), std::fscanf(b, "%63s %63s %63s", p, q, r));
      same("fscanf tokens", std::string(x) + y + z, std::string(p) + q + r);
    }
    same(std::string("last ") + closing, input_last(closing), openmx_io_last(lib, closing));
  }
  void block_fgets(const char* opening, const char* closing, int lines) const {
    std::FILE* a = input_find(opening);
    std::FILE* b = openmx_io_find(lib, opening);
    require(a && b, std::string("find ") + opening);
    for (int i = 0; i < lines; ++i) {
      char x[256], y[256];
      require(std::fgets(x, sizeof x, a) && std::fgets(y, sizeof y, b), "fgets");
      same("fgets line", std::string(x), std::string(y));
    }
    std::ungetc('\n', a);
    std::ungetc('\n', b);
    same(std::string("last after ungetc ") + closing, input_last(closing), openmx_io_last(lib, closing));
  }
};

void legacy() {
  TempDir dir;
  // The original prints its diagnostics with printf; the library writes the
  // same text to the diagnostics stream it was given.
  const std::string legacy_capture = (dir.path / "legacy-stdout.txt").string();
  const std::string library_capture = (dir.path / "library-diagnostics.txt").string();
  require(std::freopen(legacy_capture.c_str(), "w", stdout) != nullptr, "redirect stdout");
  std::FILE* diagnostics = std::fopen(library_capture.c_str(), "w");
  require(diagnostics != nullptr, "diagnostics file");
  OpenMXKeywordFiles* lib = openmx_io_keyword_files_create(14, diagnostics);
  require(lib != nullptr, "create");
  const Pair pair{lib};

  const std::string basic = dir.write("basic.dat", basic_text);
  std::string crlf_text = basic_text;
  for (std::size_t at = 0; (at = crlf_text.find('\n', at)) != std::string::npos; at += 2) crlf_text.replace(at, 1, "\r\n");
  const std::string crlf = dir.write("crlf.dat", crlf_text);
  const std::string second = dir.write("second.dat", "only.key 1\nbad.int zz\n");

  for (const std::string& path : {basic, crlf}) {
    pair.open(path);
    pair.errors("after open");
    pair.integer("scf.maxIter", 1);
    pair.integer("SCF.MAXITER", 1);
    pair.integer("scf.MaxIter ", 1);           // a trailing blank in the key never matches
    pair.integer("missing.key", 9);
    pair.integer("Tabbed.Key", 0);
    pair.integer("int.with.suffix", 0);
    pair.integer("hex.int", 0);
    pair.integer("dup.key", 0);
    pair.integer("Atoms.Number", 0);
    pair.integer("empty.value", 5);
    pair.integer("bad.int", 5);
    pair.errors("after two invalid integers");
    pair.real("scf.energycutoff", 1.0);
    pair.real("sci.double", 0.0);
    pair.real("Dos.Erange", 0.0);               // the first of two numbers
    pair.real("bad.double", 2.5);
    pair.real("empty.value", 2.5);
    pair.logical("scf.SpinPolarization", 0);
    pair.logical("logical.false", 1);
    pair.logical("bad.logical", 1);
    pair.logical("empty.value", 1);
    pair.logical("missing.key", 1);
    pair.string("System.Name", "none");
    pair.string("Mixed.case.KEY", "none");
    pair.string("empty.value", "none");
    pair.string("missing.key", "none");
    pair.choice("scf.XcType");
    pair.choice("bad.logical");
    pair.choice("empty.value");
    pair.choice("missing.key");
    pair.strings("strings.key", 3);
    pair.strings("strings.key", 4);
    pair.strings("strings.key", 1);
    pair.strings("empty.value", 1);
    pair.reals("Dos.Erange", 2);
    pair.reals("partial.vector", 3);
    pair.reals("bad.double", 1);
    pair.reals("missing.key", 2);
    pair.integers("scf.Kgrid", 3);
    pair.integers("Tabbed.Key", 1);
    pair.integers("Tabbed.Key", 2);
    pair.integers("scf.kgrid", 4);
    pair.integers("hex.int", 1);
    pair.errors("after the scalar and vector lookups");
    pair.block_fscanf("<Definition.of.Atomic.Species", "Definition.of.Atomic.Species>", 2);
    pair.block_fscanf("<Definition.of.Atomic.Species", "Wrong.Closing.Tag>", 2);
    pair.block_fscanf("<Missing.Block", "Missing.Block>", 1);
    pair.block_fgets("<Atoms.SpeciesAndCoordinates", "Atoms.SpeciesAndCoordinates>", 3);
    pair.block_fscanf("<Broken.Block", "Broken.Block>", 1);
    pair.block_fscanf("logical.false", "anything", 0);   // the last line: nothing follows
    pair.close();
  }

  // nesting: the inner file resets the counter, its errors are kept after it closes
  pair.open(basic);
  pair.integer("bad.int", 0);
  pair.errors("outer file");
  pair.open(second);
  pair.errors("inner file just opened");
  pair.integer("bad.int", 0);
  pair.integer("only.key", 0);
  pair.integer("scf.maxIter", 0);   // the outer file's keys are not visible
  pair.errors("inner file");
  pair.close();
  pair.integer("bad.int", 0);
  pair.integer("scf.maxIter", 0);
  pair.errors("outer file again");
  pair.close();
  // a file that cannot be opened (the last step: the original then holds a null stream)
  pair.open((dir.path / "does-not-exist.dat").string());

  std::fflush(stdout);
  std::fclose(diagnostics);
  same("diagnostics text", slurp(legacy_capture), slurp(library_capture));
  openmx_io_keyword_files_destroy(lib);
  std::fprintf(stderr, "PASS io legacy agreement (%zu bytes of identical diagnostics)\n", slurp(legacy_capture).size());
}

void semantics() {
  TempDir dir;
  const std::string basic = dir.write("basic.dat", basic_text);
  {
    KeywordFile file(basic);
    int n = 0;
    require(file.integer("scf.maxIter", n, 0) == Lookup::found && n == 40, "integer");
    require(file.integer("missing", n, 7) == Lookup::missing && n == 7, "missing keeps the fallback");
    require(file.integer("bad.int", n, 7) == Lookup::invalid && n == 7 && file.error_count() == 1, "invalid");
    require(file.messages().back() == "\nERROR, key=bad.int value=int\n\n", "message text");
    double d = 0;
    require(file.real("bad.double", d, 2.5) == Lookup::invalid && d == 2.5, "invalid real");
    require(file.messages().back() == "\nERROR, key=bad.double value=double\n\n", "real message text");
    std::string s;
    require(file.string("System.Name", s, "x") == Lookup::found && s == "test1", "string");
    const openmx::io::Choice choices[] = {{"lda", 1}, {"gga-pbe", 2}};
    require(file.choice("scf.XcType", n, choices) == Lookup::found && n == 2, "choice");
    require(file.choice("missing", n, choices) == Lookup::missing && n == 1, "choice fallback");
    std::string names[2];
    const std::string defaults[2] = {"a", "b"};
    require(file.strings("strings.key", names, defaults) == Lookup::found && names[1] == "beta", "strings");
    const auto species = file.block("Definition.of.Atomic.Species");
    require(species && species->size() == 2 && (*species)[0] == " H   H5.0-s2p1   H_PBE19", "block lines");
    require(!file.block("Missing.Block") && !file.block("Broken.Block"), "absent or unterminated blocks");
    require(KeywordFile::matches("abc", "ABC") && !KeywordFile::matches("ab", "abc") && !KeywordFile::matches("abc", "abd"), "matching rule");
    require(file.error_count() == 2, "error count after the block lookups");
    KeywordFile moved(std::move(file));
    require(moved.stream() != nullptr && file.stream() == nullptr && moved.error_count() == 2, "move");
    require(moved.close() == 0 && moved.stream() == nullptr && moved.close() == EOF, "explicit close");
  }
  bool threw = false;
  try { KeywordFile missing((dir.path / "none.dat").string()); } catch (const std::runtime_error&) { threw = true; }
  require(threw, "unopenable file throws");

  const std::string diag_path = (dir.path / "diag.txt").string();
  std::FILE* diag = std::fopen(diag_path.c_str(), "w");
  require(diag != nullptr, "diag");
  {
    KeywordFiles stack(2, diag);
    require(stack.depth() == 0 && stack.current() == nullptr && stack.close() == EOF, "empty stack");
    require(stack.open(basic) && stack.open(basic) && !stack.open(basic) && stack.depth() == 2, "capacity");
    require(stack.messages().back() == "input_open: can not open a further file. nestlevel=2\n", "capacity message");
    int n = 0;
    require(stack.current()->integer("bad.int", n, 0) == Lookup::invalid && stack.error_count() == 1, "stack errors");
    require(stack.close() == 0 && stack.error_count() == 1 && stack.depth() == 1, "errors survive close");
    require(stack.open(basic) && stack.error_count() == 0, "open resets");
    require(stack.close() == 0 && stack.depth() == 1, "close again");
    require(!stack.open((dir.path / "none.dat").string()) && stack.depth() == 1, "unopenable is not pushed");
    require(stack.messages().back() == "input_open: can not open " + (dir.path / "none.dat").string() + "\n", "open message");
  }
  std::fclose(diag);
  const std::string text = slurp(diag_path);
  require(text.find("nestlevel=2") != std::string::npos && text.find("ERROR, key=bad.int value=int") != std::string::npos, "diagnostics stream");
  std::fprintf(stderr, "PASS io semantics\n");
}

void invalid() {
  TempDir dir;
  const std::string basic = dir.write("basic.dat", basic_text);
  int n = 0;
  double d = 0;
  char text[4];
  require(openmx_io_int(nullptr, "k", &n, 1) == -1 && openmx_io_double(nullptr, "k", &d, 1) == -1, "null stack");
  require(openmx_io_find(nullptr, "k") == nullptr && openmx_io_last(nullptr, "k") == 0, "null stack blocks");
  require(openmx_io_depth(nullptr) == 0 && openmx_io_error_count(nullptr) == 0 && openmx_io_close(nullptr) == EOF, "null stack state");
  OpenMXKeywordFiles* files = openmx_io_keyword_files_create(0, nullptr);
  require(files != nullptr, "create with a capacity below one");
  require(openmx_io_int(files, "k", &n, 1) == -1 && openmx_io_open(files, nullptr) == 0, "nothing open");
  require(openmx_io_open(files, basic.c_str()) == 1 && openmx_io_open(files, basic.c_str()) == 0, "capacity one");
  require(openmx_io_int(files, nullptr, &n, 1) == -1 && openmx_io_int(files, "scf.maxIter", nullptr, 1) == -1, "null arguments");
  require(openmx_io_string(files, "System.Name", text, sizeof text, "fallback") == -1 && std::strcmp(text, "fal") == 0, "value too long: the fallback, cut to the capacity");
  require(openmx_io_string(files, "System.Name", text, 0, "x") == -1, "zero capacity");
  require(openmx_io_string2int(files, "scf.XcType", &n, 0, nullptr, nullptr) == -1, "no choices");
  require(openmx_io_doublev(files, "Dos.Erange", 2, nullptr, nullptr) == -1, "null vectors");
  require(openmx_io_doublev(files, "Dos.Erange", 0, nullptr, nullptr) == 1, "zero-length vector of a present key");
  require(openmx_io_intv(files, "missing", 0, nullptr, nullptr) == 0, "zero-length vector of a missing key");
  require(openmx_io_error_count(files) == 0, "argument errors are not file errors");
  require(openmx_io_cmpstring(nullptr, &n, 1, nullptr, nullptr) == 0, "cmpstring without input");
  openmx_io_keyword_files_destroy(files);
  std::fprintf(stderr, "PASS io invalid\n");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string name = argc > 1 ? argv[1] : "legacy";
    if (name == "legacy") legacy();
    else if (name == "semantics") semantics();
    else if (name == "invalid") invalid();
    else throw std::runtime_error("unknown io test case");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
