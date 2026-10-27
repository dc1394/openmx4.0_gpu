#include <openmx/basis/pao_file.hpp>
#include <openmx/basis/radial_table.hpp>

#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace openmx::basis {
namespace {
bool space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v'; }
bool equal(std::string_view a, std::string_view b)
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
    throw std::runtime_error("PAO "+section_+" at line "+std::to_string(line_)+": "+reason);
  }
 private:
  std::string_view text_;
  std::string section_;
  std::size_t line_;
};

class Document {
 public:
  explicit Document(std::istream& input) : text_(std::istreambuf_iterator<char>(input),{}) {
    if (input.bad()) throw std::runtime_error("PAO input read failed");
    if (text_.find('\0') != std::string::npos) throw std::runtime_error("PAO input contains a NUL byte");
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
      throw std::runtime_error("PAO missing scalar "+std::string(key));
    }
    Tokens tokens(lines_[*index].text,std::string(key),*index+1);
    tokens.next();
    return tokens.number<Number>();
  }
  Tokens section(const std::string& name) const {
    const auto begin = find("<"+name);
    if (!begin) throw std::runtime_error("PAO missing section "+name);
    const auto end = find(name+">",*begin+1);
    if (!end) throw std::runtime_error("PAO missing closing tag for "+name);
    const auto offset = lines_[*begin+1].offset;
    return {std::string_view(text_).substr(offset,lines_[*end].offset-offset),name,*begin+2};
  }
  Tokens after(std::size_t line, const std::string& name) const {
    const auto offset = line+1 < lines_.size() ? lines_[line+1].offset : text_.size();
    return {std::string_view(text_).substr(offset),name,line+2};
  }
 private:
  struct Line { std::string_view text; std::size_t offset; };
  std::string text_;
  std::vector<Line> lines_;
};

std::size_t checked_product(std::size_t a, std::size_t b)
{
  if (b && a > std::vector<double>().max_size()/b)
    throw std::length_error("PAO table dimensions overflow");
  return a*b;
}

void read_mesh_point(Tokens& input, std::size_t i,
                     std::vector<double>& logarithms, std::vector<double>& radii)
{
  logarithms[i] = input.number<double>();
  radii[i] = input.number<double>();
  if (radii[i] <= 0 || (i && (radii[i] <= radii[i-1] || logarithms[i] <= logarithms[i-1])))
    input.fail("mesh must be positive and strictly increasing");
}
} // namespace

int PaoMetadata::atomic_number() const
{
  if (!std::isfinite(atom_species) || atom_species < 0 || atom_species > INT_MAX)
    throw std::invalid_argument("invalid PAO atomic species");
  return atom_species > 1e-15 && atom_species < 1 ? 1 : static_cast<int>(atom_species);
}

PaoFile PaoFile::read(std::istream& input, PaoReadMode mode)
{
  if (mode != PaoReadMode::full && mode != PaoReadMode::metadata)
    throw std::invalid_argument("invalid PAO read mode");
  const Document document(input);
  PaoFile result;
  auto& m = result.metadata_;
  m.atom_species = document.scalar<double>("AtomSpecies");
  m.mesh_points = document.scalar<int>("grid.num.output");
  m.cutoff = document.scalar<double>("radial.cutoff.pao",5.);
  m.lmax = document.scalar<int>("PAO.Lmax");
  m.multiplicity = document.scalar<int>("PAO.Mul");
  if (m.atom_species < 0 || m.atom_species > INT_MAX || m.mesh_points < 3 ||
      m.mesh_points > INT_MAX-2 || m.lmax < 0 || m.lmax > 29 || m.multiplicity < 1 || m.cutoff <= 0)
    throw std::invalid_argument("PAO invalid species, dimensions or cutoff");
  const auto channels = checked_product(static_cast<std::size_t>(m.lmax+1),m.multiplicity);
  if (const auto line = document.find("Lmax=")) {
    if (channels > document.size()/5)
      throw std::invalid_argument("PAO energy dimensions exceed the input size");
    // Preserve the data set's labels, without retaining any input stream.
    auto tokens = document.after(*line,"orbital energies");
    result.energies_.assign(channels,std::numeric_limits<double>::quiet_NaN());
    for (std::size_t channel = 0; channel < channels; ++channel) {
      if (!equal(tokens.next(),"l") || !equal(tokens.next(),"mu")) tokens.fail("expected l mu energy labels");
      const int l = tokens.number<int>(), mu = tokens.number<int>();
      if (l < 0 || l > m.lmax || mu < 0 || mu >= m.multiplicity) tokens.fail("invalid energy channel");
      auto& energy = result.energies_[static_cast<std::size_t>(l)*m.multiplicity+mu];
      if (std::isfinite(energy)) tokens.fail("duplicate energy channel");
      energy = tokens.number<double>();
    }
  }
  if (mode == PaoReadMode::metadata) return result;

  const auto orbital_size = checked_product(channels,m.mesh_points);
  if (orbital_size > document.size() || static_cast<std::size_t>(m.mesh_points) > document.size()/3)
    throw std::invalid_argument("PAO radial dimensions exceed the input size");

  result.log_radii_.resize(m.mesh_points);
  result.radii_.resize(m.mesh_points);
  result.density_.resize(m.mesh_points);
  auto density = document.section("valence.charge.density");
  for (std::size_t i = 0; i < result.radii_.size(); ++i) {
    read_mesh_point(density,i,result.log_radii_,result.radii_);
    result.density_[i] = density.number<double>();
  }
  density.end();
  result.orbitals_.resize(orbital_size);
  std::vector<double> log_mesh(m.mesh_points), mesh(m.mesh_points);
  for (int l = 0; l <= m.lmax; ++l) {
    auto table = document.section("pseudo.atomic.orbitals.L="+std::to_string(l));
    for (std::size_t i = 0; i < mesh.size(); ++i) {
      read_mesh_point(table,i,log_mesh,mesh);
      // The file may print the same mesh at different decimal precisions.
      if (std::abs(mesh[i]-result.radii_[i]) > 1e-10*(1+std::abs(result.radii_[i])) ||
          std::abs(log_mesh[i]-result.log_radii_[i]) > 1e-10*(1+std::abs(result.log_radii_[i])))
        table.fail("density and orbital meshes disagree");
      for (int mu = 0; mu < m.multiplicity; ++mu)
        result.orbitals_[(static_cast<std::size_t>(l)*m.multiplicity+mu)*mesh.size()+i] = table.number<double>();
    }
    table.end();
    // Legacy Read_PAO uses the coordinates of the last orbital block.
    result.log_radii_ = log_mesh;
    result.radii_ = mesh;
  }
  return result;
}

PaoFile PaoFile::from_file(const std::string& path, PaoReadMode mode)
{
  std::ifstream input(path,std::ios::binary);
  if (!input) throw std::runtime_error("cannot open PAO file: "+path);
  try { return read(input,mode); }
  catch (const std::exception& error) { throw std::runtime_error(path+": "+error.what()); }
}

std::size_t PaoFile::channel_index(int angular, int multiplicity) const
{
  if (angular < 0 || angular > metadata_.lmax || multiplicity < 0 || multiplicity >= metadata_.multiplicity)
    throw std::out_of_range("PAO channel index");
  return static_cast<std::size_t>(angular)*metadata_.multiplicity+multiplicity;
}

std::span<const double> PaoFile::orbital(int angular, int multiplicity) const
{
  const auto channel = channel_index(angular,multiplicity);
  if (!has_radial_data()) throw std::logic_error("PAO metadata-only file has no orbitals");
  return {orbitals_.data()+channel*radii_.size(),radii_.size()};
}

double PaoFile::sample(int angular, int multiplicity, double radius) const
{
  return sample_radial_row(RadialKind::pao,angular,radii_,orbital(angular,multiplicity),0,radius);
}

std::optional<double> PaoFile::orbital_energy(int angular, int multiplicity) const
{
  const auto channel = channel_index(angular,multiplicity);
  if (energies_.empty()) return std::nullopt;
  return energies_[channel];
}

} // namespace openmx::basis
