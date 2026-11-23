#include <openmx/exx/exchange.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace openmx::exx {
namespace {

// The arrays of a system must fit together, and both neighbour lists of an
// atom must begin with the atom itself: the code below takes the first
// columns of the T blocks of an atom as the columns of that atom.
void check(const System& system)
{
  const std::size_t atoms = system.orbital_count.size();
  if (atoms == 0 || system.abf_count.size() != atoms || system.c_neighbours.size() != atoms ||
      system.v_neighbours.size() != atoms || system.c_offset.size() != atoms || system.t.size() != atoms ||
      system.v.size() != atoms)
    throw std::invalid_argument("the arrays of a system do not agree in the number of atoms");
  const auto valid = [atoms](const Site& site) {
    return site.atom >= 0 && static_cast<std::size_t>(site.atom) < atoms;
  };
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::vector<Site>& c = system.c_neighbours[a];
    const std::vector<Site>& v = system.v_neighbours[a];
    const Site self{static_cast<int>(a),{0,0,0}};
    if (c.empty() || v.empty() || !(c.front() == self) || !(v.front() == self))
      throw std::invalid_argument("the neighbour lists of an atom must begin with the atom itself");
    const std::vector<std::size_t>& offset = system.c_offset[a];
    if (offset.size() != c.size()+1 || offset.front() != 0)
      throw std::invalid_argument("the column offsets of the T blocks do not match the C neighbours");
    for (std::size_t i = 0; i < c.size(); ++i)
      if (!valid(c[i]) || offset[i+1] < offset[i] ||
          offset[i+1]-offset[i] != system.orbital_count[static_cast<std::size_t>(c[i].atom)])
        throw std::invalid_argument("a T block does not match the orbitals of its C neighbour");
    if (system.t[a].size() != system.abf_count[a]*system.orbital_count[a]*offset.back())
      throw std::invalid_argument("the T blocks of an atom have the wrong size");
    if (system.v[a].size() != v.size()) throw std::invalid_argument("the V blocks do not match the V neighbours");
    for (std::size_t i = 0; i < v.size(); ++i)
      if (!valid(v[i]) || system.v[a][i].size() != system.abf_count[a]*system.abf_count[static_cast<std::size_t>(v[i].atom)])
        throw std::invalid_argument("a V block does not match the ABFs of its V neighbour");
  }
}

std::vector<std::size_t> orbital_offsets(const System& system)
{
  std::vector<std::size_t> offset(1,0);
  for (const std::size_t u : system.orbital_count) offset.push_back(offset.back()+u);
  return offset;
}

// For every column l of the T blocks of atom b: the orbital of the unit cell
// that the column belongs to.  Lattice images of an atom give the same
// orbitals.
std::vector<std::size_t> folded_columns(const System& system, const std::vector<std::size_t>& offset, std::size_t b)
{
  std::vector<std::size_t> folded;
  folded.reserve(system.c_offset[b].back());
  for (const Site& site : system.c_neighbours[b]) {
    const std::size_t c = static_cast<std::size_t>(site.atom);
    for (std::size_t lambda = 0; lambda < system.orbital_count[c]; ++lambda) folded.push_back(offset[c]+lambda);
  }
  return folded;
}

// The contribution of the centre atom a, section 6.4.  The columns of the
// density matrix are numbered 0..s-1 by the caller: the orbitals of the sites
// of S in the site form, the orbitals of the unit cell in the Gamma form.
// dloc is D[loc(A), columns], width(a) x s; its first u_a rows belong to A.
//
// The sink hands out the blocks and receives the results.  Its groups are
// the V neighbours of the centre: one site each in the site form, all images
// of one atom together in the Gamma form.
//   groups()        the number of groups
//   atom(g)         the atom b of group g
//   v(g)            its V block [P, Q]
//   t(x), width(x)  the T blocks of atom x, [(P,mu), l] with width(x) columns
//   columns(g)      the column of the density matrix of every column l of
//                   t(atom(g)); the first u_b of them belong to B itself
//   first(g, t1)    rows A,      columns B        term (i)
//   second(g, t2)   rows A,      columns loc(B)   term (ii)
//   last(t4)        rows loc(A), all columns      terms (iii) and (iv)
// Term (iii) is added to the bracket of term (iv) at the columns of B, so one
// product with the T blocks transforms both.
//
// Layouts, with P and mu on A, Q and nu on B, l a column of loc(B):
//   T_a [(P,mu), lambda]   G [(P,mu), column]   W [(P,nu), l]   Wt [(P,l), nu]
// Every sum over P together with another index is one matrix product.
template <class Sink>
void centre_contribution(const System& system, std::size_t a, std::size_t s, const std::vector<double>& dloc,
                         Sink& sink, const DenseBackend& backend)
{
  const std::size_t np = system.abf_count[a], ua = system.orbital_count[a], ell = sink.width(a);
  const std::size_t rows = np*ua;
  const double* tst = sink.t(a);
  const double* da = dloc.data();

  std::vector<double> g(rows*s), bracket(rows*s,0.0);
  backend.multiply(rows,s,ell,1.0,tst,dloc.data(),0.0,g.data());      // G = T D[loc(A), .]

  std::vector<double> w, wt, gathered, density, product;
  for (std::size_t ib = 0; ib < sink.groups(); ++ib) {
    const std::size_t b = sink.atom(ib);
    const std::size_t nq = system.abf_count[b], ub = system.orbital_count[b], lb = sink.width(b);
    const std::vector<std::size_t>& column = sink.columns(ib);         // lb entries

    // W_{P,B} = sum_Q V_PQ T_Q, equation (10), and its copy with l before nu.
    w.resize(np*ub*lb);
    backend.multiply(np,ub*lb,nq,1.0,sink.v(ib),sink.t(b),0.0,w.data());
    wt.resize(np*lb*ub);
    for (std::size_t p = 0; p < np; ++p)
      for (std::size_t nu = 0; nu < ub; ++nu)
        for (std::size_t l = 0; l < lb; ++l) wt[(p*lb+l)*ub+nu] = w[(p*ub+nu)*lb+l];

    // (i)  sum_P G_P[., loc(B)] W_P^T:  [mu,(P,l)] [(P,l),nu]
    gathered.resize(ua*np*lb);
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t p = 0; p < np; ++p) {
        const double* source = g.data()+(p*ua+mu)*s;
        double* target = gathered.data()+(mu*np+p)*lb;
        for (std::size_t l = 0; l < lb; ++l) target[l] = source[column[l]];
      }
    product.resize(ua*ub);
    backend.multiply(ua,ub,np*lb,1.0,gathered.data(),wt.data(),0.0,product.data());
    sink.first(ib,product.data());

    // (ii)  sum_P G_P[., B] W_P:  [mu,(P,nu)] [(P,nu),l]
    gathered.resize(ua*np*ub);
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t p = 0; p < np; ++p) {
        const double* source = g.data()+(p*ua+mu)*s;
        double* target = gathered.data()+(mu*np+p)*ub;
        for (std::size_t nu = 0; nu < ub; ++nu) target[nu] = source[column[nu]];
      }
    product.resize(ua*lb);
    backend.multiply(ua,lb,np*ub,1.0,gathered.data(),w.data(),0.0,product.data());
    sink.second(ib,product.data());

    // (iii)  bracket[., B] += D[A, loc(B)] W_P^T:  [(P,nu),l] [mu,l]^T
    density.resize(ua*lb);
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t l = 0; l < lb; ++l) density[mu*lb+l] = da[mu*s+column[l]];
    product.resize(np*ub*ua);
    backend.multiply_transposed(np*ub,ua,lb,1.0,w.data(),density.data(),0.0,product.data());
    for (std::size_t p = 0; p < np; ++p)
      for (std::size_t nu = 0; nu < ub; ++nu)
        for (std::size_t mu = 0; mu < ua; ++mu) bracket[(p*ua+mu)*s+column[nu]] += product[(p*ub+nu)*ua+mu];

    // (iv)  bracket[., loc(B)] += D[A, B] W_P:  [(P,l),nu] [mu,nu]^T
    density.resize(ua*ub);
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t nu = 0; nu < ub; ++nu) density[mu*ub+nu] = da[mu*s+column[nu]];
    product.resize(np*lb*ua);
    backend.multiply_transposed(np*lb,ua,ub,1.0,wt.data(),density.data(),0.0,product.data());
    for (std::size_t p = 0; p < np; ++p)
      for (std::size_t l = 0; l < lb; ++l)
        for (std::size_t mu = 0; mu < ua; ++mu) bracket[(p*ua+mu)*s+column[l]] += product[(p*lb+l)*ua+mu];
  }

  // rows loc(A):  sum_P T_P^T bracket_P
  product.resize(ell*s);
  backend.transposed_multiply(ell,s,rows,1.0,tst,bracket.data(),0.0,product.data());
  sink.last(product.data());
}

// ---- site form -------------------------------------------------------------

std::array<int,3> sum(const std::array<int,3>& a, const std::array<int,3>& b)
{
  return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};
}

std::array<int,3> difference(const std::array<int,3>& a, const std::array<int,3>& b)
{
  return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}

class SiteSink {
 public:
  SiteSink(const System& system, std::size_t a, double factor, BlockMatrix& k)
      : system_(system), a_(a), factor_(factor), k_(k), offset_(1,0)
  {
    // S: the sites of loc(B) for all V neighbours B, in the order of their
    // first appearance.
    for (const Site& b : system.v_neighbours[a])
      for (const Site& c : system.c_neighbours[static_cast<std::size_t>(b.atom)]) {
        const Site site{c.atom,sum(b.cell,c.cell)};
        if (index_.emplace(key(site),sites_.size()).second) {
          sites_.push_back(site);
          offset_.push_back(offset_.back()+system.orbital_count[static_cast<std::size_t>(c.atom)]);
        }
      }
  }

  std::size_t size() const noexcept { return offset_.back(); }

  // Every V neighbour is a group of its own, with the blocks as they are stored.
  std::size_t groups() const noexcept { return system_.v_neighbours[a_].size(); }
  std::size_t atom(std::size_t ib) const { return static_cast<std::size_t>(system_.v_neighbours[a_][ib].atom); }
  const double* v(std::size_t ib) const { return system_.v[a_][ib].data(); }
  const double* t(std::size_t atom) const { return system_.t[atom].data(); }
  std::size_t width(std::size_t atom) const { return system_.c_offset[atom].back(); }

  // D[loc(A), S]
  std::vector<double> density(const DensityBlocks& blocks) const
  {
    const std::vector<Site>& loc = system_.c_neighbours[a_];
    const std::size_t s = size();
    std::vector<double> dloc(system_.c_offset[a_].back()*s), block;
    for (std::size_t i = 0; i < loc.size(); ++i) {
      const std::size_t rows = system_.orbital_count[static_cast<std::size_t>(loc[i].atom)];
      for (std::size_t j = 0; j < sites_.size(); ++j) {
        const std::size_t columns = offset_[j+1]-offset_[j];
        block.assign(rows*columns,0.0);
        blocks(BlockKey{loc[i].atom,sites_[j].atom,difference(sites_[j].cell,loc[i].cell)},block);
        for (std::size_t r = 0; r < rows; ++r)
          std::copy_n(block.begin()+static_cast<std::ptrdiff_t>(r*columns),columns,
                      dloc.begin()+static_cast<std::ptrdiff_t>((system_.c_offset[a_][i]+r)*s+offset_[j]));
      }
    }
    return dloc;
  }

  const std::vector<std::size_t>& columns(std::size_t ib)
  {
    const Site& b = system_.v_neighbours[a_][ib];
    columns_.clear();
    for (const Site& c : system_.c_neighbours[static_cast<std::size_t>(b.atom)]) {
      const std::size_t j = index_.at(key(Site{c.atom,sum(b.cell,c.cell)}));
      for (std::size_t column = offset_[j]; column < offset_[j+1]; ++column) columns_.push_back(column);
    }
    return columns_;
  }

  void first(std::size_t ib, const double* values)
  {
    const Site& b = system_.v_neighbours[a_][ib];
    const std::span<double> block = k_.at(BlockKey{static_cast<int>(a_),b.atom,b.cell});
    for (std::size_t e = 0; e < block.size(); ++e) block[e] += factor_*values[e];
  }

  void second(std::size_t ib, const double* values)
  {
    const Site& b = system_.v_neighbours[a_][ib];
    const std::size_t atom = static_cast<std::size_t>(b.atom);
    const std::size_t ua = system_.orbital_count[a_], lb = system_.c_offset[atom].back();
    const std::vector<Site>& loc = system_.c_neighbours[atom];
    for (std::size_t j = 0; j < loc.size(); ++j) {
      const std::size_t begin = system_.c_offset[atom][j], width = system_.c_offset[atom][j+1]-begin;
      const std::span<double> block = k_.at(BlockKey{static_cast<int>(a_),loc[j].atom,sum(b.cell,loc[j].cell)});
      for (std::size_t mu = 0; mu < ua; ++mu)
        for (std::size_t lambda = 0; lambda < width; ++lambda)
          block[mu*width+lambda] += factor_*values[mu*lb+begin+lambda];
    }
  }

  void last(const double* values)
  {
    const std::vector<Site>& loc = system_.c_neighbours[a_];
    const std::size_t s = size();
    for (std::size_t i = 0; i < loc.size(); ++i) {
      const std::size_t begin = system_.c_offset[a_][i], rows = system_.c_offset[a_][i+1]-begin;
      for (std::size_t j = 0; j < sites_.size(); ++j) {
        const std::size_t columns = offset_[j+1]-offset_[j];
        const std::span<double> block = k_.at(BlockKey{loc[i].atom,sites_[j].atom,difference(sites_[j].cell,loc[i].cell)});
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t c = 0; c < columns; ++c) block[r*columns+c] += factor_*values[(begin+r)*s+offset_[j]+c];
      }
    }
  }

 private:
  static std::array<int,4> key(const Site& site) { return {site.atom,site.cell[0],site.cell[1],site.cell[2]}; }

  const System& system_;
  std::size_t a_;
  double factor_;
  BlockMatrix& k_;
  std::vector<Site> sites_;
  std::vector<std::size_t> offset_;                 // first column of every site of S
  std::map<std::array<int,4>,std::size_t> index_;   // position of a site in S
  std::vector<std::size_t> columns_;
};

// ---- Gamma form ------------------------------------------------------------

// The sites fold onto the atoms of the unit cell: all images of an atom
// address the same orbitals.  The contribution of a centre is linear in the
// V block of a neighbour and in the columns of the T blocks, so the folding
// is done before the products.  The V blocks of the images of an atom are
// added up, and so are the columns of the T blocks that belong to one
// orbital.  A cell that is smaller than the cutoffs then costs five products
// for every atom among the V neighbours instead of five for every image, and
// no array has more than N columns.
class GammaSink {
 public:
  GammaSink(const System& system, std::size_t a, double factor, std::span<double> k,
            const std::vector<std::size_t>& offset)
      : system_(system), a_(a), factor_(factor), k_(k), offset_(offset), n_(offset.back()),
        folded_(system.orbital_count.size())
  {
    // The groups: the atoms among the V neighbours, in the order of their
    // first appearance.
    std::vector<std::size_t> group(system.orbital_count.size(),none);
    for (std::size_t ib = 0; ib < system.v_neighbours[a].size(); ++ib) {
      const std::size_t b = static_cast<std::size_t>(system.v_neighbours[a][ib].atom);
      const std::vector<double>& block = system.v[a][ib];
      if (group[b] == none) {
        group[b] = atoms_.size();
        atoms_.push_back(b);
        v_.push_back(block);
      } else {
        std::vector<double>& sum = v_[group[b]];
        for (std::size_t e = 0; e < block.size(); ++e) sum[e] += block[e];
      }
    }
  }

  std::size_t groups() const noexcept { return atoms_.size(); }
  std::size_t atom(std::size_t g) const { return atoms_[g]; }
  const double* v(std::size_t g) const { return v_[g].data(); }
  const double* t(std::size_t atom)
  {
    const Folded& f = folded(atom);
    return f.t.empty() ? system_.t[atom].data() : f.t.data();
  }
  std::size_t width(std::size_t atom) { return folded(atom).column.size(); }

  // The orbitals of loc(A): the rows of D[loc(A), .] and of the last result.
  const std::vector<std::size_t>& rows() { return folded(a_).column; }
  const std::vector<std::size_t>& columns(std::size_t g) { return folded(atoms_[g]).column; }

  void first(std::size_t g, const double* values)
  {
    const std::size_t b = atoms_[g], ua = system_.orbital_count[a_], ub = system_.orbital_count[b];
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t nu = 0; nu < ub; ++nu) k_[(offset_[a_]+mu)*n_+offset_[b]+nu] += factor_*values[mu*ub+nu];
  }

  void second(std::size_t g, const double* values)
  {
    const std::vector<std::size_t>& column = folded(atoms_[g]).column;
    const std::size_t ua = system_.orbital_count[a_], lb = column.size();
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t l = 0; l < lb; ++l) k_[(offset_[a_]+mu)*n_+column[l]] += factor_*values[mu*lb+l];
  }

  void last(const double* values)
  {
    const std::vector<std::size_t>& row = folded(a_).column;
    for (std::size_t lambda = 0; lambda < row.size(); ++lambda)
      for (std::size_t column = 0; column < n_; ++column) k_[row[lambda]*n_+column] += factor_*values[lambda*n_+column];
  }

 private:
  static constexpr std::size_t none = static_cast<std::size_t>(-1);

  // The T blocks of an atom with one column for every orbital of loc(atom).
  struct Folded {
    std::vector<std::size_t> column;   // the orbital of every column
    std::vector<double> t;             // empty if no two C neighbours are images of one atom:
                                       // the stored T blocks have these columns already
  };

  // Made on first use: one centre needs its own blocks and those of its V
  // neighbours only.
  const Folded& folded(std::size_t x)
  {
    Folded& f = folded_[x];
    if (!f.column.empty()) return f;
    const std::vector<Site>& loc = system_.c_neighbours[x];
    std::vector<std::size_t> begin(system_.orbital_count.size(),none);   // first column of every atom
    bool images = false;
    for (const Site& site : loc) {
      const std::size_t c = static_cast<std::size_t>(site.atom);
      if (begin[c] != none) {
        images = true;
        continue;
      }
      begin[c] = f.column.size();
      for (std::size_t lambda = 0; lambda < system_.orbital_count[c]; ++lambda) f.column.push_back(offset_[c]+lambda);
    }
    if (images) {
      const std::size_t rows = system_.abf_count[x]*system_.orbital_count[x];
      const std::size_t ell = system_.c_offset[x].back(), width = f.column.size();
      f.t.assign(rows*width,0.0);
      for (std::size_t row = 0; row < rows; ++row) {
        const double* source = system_.t[x].data()+row*ell;
        double* target = f.t.data()+row*width;
        for (std::size_t i = 0; i < loc.size(); ++i) {
          const std::size_t c = static_cast<std::size_t>(loc[i].atom);
          const std::size_t from = system_.c_offset[x][i];
          for (std::size_t lambda = 0; lambda < system_.orbital_count[c]; ++lambda)
            target[begin[c]+lambda] += source[from+lambda];
        }
      }
    }
    return f;
  }

  const System& system_;
  std::size_t a_;
  double factor_;
  std::span<double> k_;
  const std::vector<std::size_t>& offset_;
  std::size_t n_;
  std::vector<std::size_t> atoms_;         // the atom of every group
  std::vector<std::vector<double>> v_;     // sum of the V blocks of the images of that atom
  std::vector<Folded> folded_;
};

// D[loc(A), .] of a dense matrix: the rows of the orbitals of loc(A).
std::vector<double> gathered_rows(std::span<const double> density, std::size_t n, const std::vector<std::size_t>& row)
{
  std::vector<double> dloc(row.size()*n);
  for (std::size_t lambda = 0; lambda < row.size(); ++lambda)
    std::copy_n(density.begin()+static_cast<std::ptrdiff_t>(row[lambda]*n),n,
                dloc.begin()+static_cast<std::ptrdiff_t>(lambda*n));
  return dloc;
}

void check_dense(std::size_t n, std::size_t size, const char* what)
{
  if (size != n*n) throw std::invalid_argument(what);
}

} // namespace

// ---- site form -------------------------------------------------------------

BlockMatrix::BlockMatrix(std::vector<std::size_t> orbital_count) : orbital_count_(std::move(orbital_count)) {}

std::span<double> BlockMatrix::at(const BlockKey& key)
{
  const std::size_t atoms = orbital_count_.size();
  if (key.row < 0 || key.column < 0 || static_cast<std::size_t>(key.row) >= atoms ||
      static_cast<std::size_t>(key.column) >= atoms)
    throw std::out_of_range("an atom of a block does not exist");
  const auto found = blocks_.find(key);
  if (found != blocks_.end()) return found->second;
  const std::size_t size = orbital_count_[static_cast<std::size_t>(key.row)]*orbital_count_[static_cast<std::size_t>(key.column)];
  return blocks_.emplace(key,std::vector<double>(size,0.0)).first->second;
}

std::span<const double> BlockMatrix::find(const BlockKey& key) const noexcept
{
  const auto found = blocks_.find(key);
  return found != blocks_.end() ? std::span<const double>(found->second) : std::span<const double>();
}

void BlockMatrix::scale(double factor) noexcept
{
  for (auto& [key, block] : blocks_)
    for (double& value : block) value *= factor;
}

void add_exchange_blocks(const System& system, std::size_t atom, double factor, const DensityBlocks& density,
                         BlockMatrix& k, const DenseBackend& backend)
{
  check(system);
  if (atom >= system.orbital_count.size()) throw std::out_of_range("the centre atom does not exist");
  if (!density) throw std::invalid_argument("the exchange matrix needs a density matrix");
  if (!std::ranges::equal(k.orbital_count(),system.orbital_count))
    throw std::invalid_argument("the blocks of K do not belong to this system");
  SiteSink sink(system,atom,factor,k);
  centre_contribution(system,atom,sink.size(),sink.density(density),sink,backend);
}

BlockMatrix exchange_blocks(const System& system, const DensityBlocks& density, double alpha,
                            const DenseBackend& backend)
{
  BlockMatrix k(system.orbital_count);
  for (std::size_t a = 0; a < system.orbital_count.size(); ++a) add_exchange_blocks(system,a,-alpha,density,k,backend);
  return k;
}

double exchange_energy(const DensityBlocks& density, const BlockMatrix& k)
{
  if (!density) throw std::invalid_argument("the exchange energy needs a density matrix");
  double total = 0.0;
  std::vector<double> block;
  for (const auto& [key, values] : k.blocks()) {
    block.assign(values.size(),0.0);
    density(key,block);
    for (std::size_t e = 0; e < values.size(); ++e) total += block[e]*values[e];
  }
  return 0.5*total;
}

double exchange_change(const DensityBlocks& density, const BlockMatrix& k, const BlockMatrix& k_previous)
{
  if (!density) throw std::invalid_argument("the change of the exchange matrix needs a density matrix");
  if (!std::ranges::equal(k.orbital_count(),k_previous.orbital_count()))
    throw std::invalid_argument("the two exchange matrices belong to different systems");
  double total = 0.0;
  std::vector<double> block;
  for (const auto& [key, values] : k.blocks()) {
    const std::span<const double> previous = k_previous.find(key);
    block.assign(values.size(),0.0);
    density(key,block);
    for (std::size_t e = 0; e < values.size(); ++e) total += block[e]*(values[e]-(previous.empty() ? 0.0 : previous[e]));
  }
  for (const auto& [key, values] : k_previous.blocks()) {
    if (!k.find(key).empty()) continue;
    block.assign(values.size(),0.0);
    density(key,block);
    for (std::size_t e = 0; e < values.size(); ++e) total -= block[e]*values[e];
  }
  return total;
}

// ---- Gamma form ------------------------------------------------------------

std::size_t orbital_total(const System& system) noexcept
{
  std::size_t total = 0;
  for (const std::size_t u : system.orbital_count) total += u;
  return total;
}

void add_exchange_gamma(const System& system, std::size_t atom, double factor, std::span<const double> density,
                        std::span<double> k, const DenseBackend& backend)
{
  check(system);
  if (atom >= system.orbital_count.size()) throw std::out_of_range("the centre atom does not exist");
  const std::vector<std::size_t> offset = orbital_offsets(system);
  const std::size_t n = offset.back();
  check_dense(n,density.size(),"the density matrix is not N x N");
  check_dense(n,k.size(),"the exchange matrix is not N x N");
  GammaSink sink(system,atom,factor,k,offset);
  centre_contribution(system,atom,n,gathered_rows(density,n,sink.rows()),sink,backend);
}

std::vector<double> exchange_gamma(const System& system, std::span<const double> density, double alpha,
                                   const DenseBackend& backend)
{
  const std::size_t n = orbital_total(system);
  std::vector<double> k(n*n,0.0);
  for (std::size_t a = 0; a < system.orbital_count.size(); ++a) add_exchange_gamma(system,a,-alpha,density,k,backend);
  return k;
}

double exchange_energy_gamma(std::span<const double> density, std::span<const double> k)
{
  if (density.size() != k.size()) throw std::invalid_argument("the density and exchange matrices differ in size");
  double total = 0.0;
  for (std::size_t e = 0; e < k.size(); ++e) total += density[e]*k[e];
  return 0.5*total;
}

double exchange_change_gamma(std::span<const double> density, std::span<const double> k,
                             std::span<const double> k_previous)
{
  if (density.size() != k.size() || k.size() != k_previous.size())
    throw std::invalid_argument("the density and exchange matrices differ in size");
  double total = 0.0;
  for (std::size_t e = 0; e < k.size(); ++e) total += density[e]*(k[e]-k_previous[e]);
  return total;
}

ExchangeGradient exchange_gradient_gamma(const System& system, std::span<const double> density, double alpha,
                                         const DenseBackend& backend)
{
  check(system);
  const std::size_t atoms = system.orbital_count.size();
  const std::vector<std::size_t> offset = orbital_offsets(system);
  const std::size_t n = offset.back();
  check_dense(n,density.size(),"the density matrix is not N x N");
  std::vector<std::vector<std::size_t>> folded(atoms);
  for (std::size_t a = 0; a < atoms; ++a) folded[a] = folded_columns(system,offset,a);

  // G_P = T_P D[loc(A), .] for every atom, [(P,mu), column].
  std::vector<std::vector<double>> g(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t rows = system.abf_count[a]*system.orbital_count[a];
    g[a].resize(rows*n);
    backend.multiply(rows,n,folded[a].size(),1.0,system.t[a].data(),gathered_rows(density,n,folded[a]).data(),0.0,
                     g[a].data());
  }

  ExchangeGradient gradient;
  gradient.t.resize(atoms);
  gradient.v.resize(atoms);
  std::vector<double> w, h1, h2, bracket, columns_of_density;
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t np = system.abf_count[a], ua = system.orbital_count[a], la = folded[a].size();
    const std::size_t rows = np*ua;
    const double* ta = system.t[a].data();
    bracket.assign(rows*n,0.0);                                        // R_P + R'_P of equation (16)
    gradient.v[a].resize(system.v_neighbours[a].size());
    for (std::size_t ib = 0; ib < system.v_neighbours[a].size(); ++ib) {
      const std::size_t b = static_cast<std::size_t>(system.v_neighbours[a][ib].atom);
      const std::size_t nq = system.abf_count[b], ub = system.orbital_count[b], lb = folded[b].size();
      const double* tb = system.t[b].data();
      const std::vector<std::size_t>& cb = folded[b];
      const std::vector<std::size_t>& ca = folded[a];

      w.resize(np*ub*lb);                                              // W [(P,nu), l]
      backend.multiply(np,ub*lb,nq,1.0,system.v[a][ib].data(),tb,0.0,w.data());
      for (std::size_t p = 0; p < np; ++p)
        for (std::size_t mu = 0; mu < ua; ++mu) {
          const double* row = density.data()+(offset[a]+mu)*n;         // D[A, .]
          double* target = bracket.data()+(p*ua+mu)*n;
          for (std::size_t nu = 0; nu < ub; ++nu) {
            const double* source = w.data()+(p*ub+nu)*lb;
            const double d = row[offset[b]+nu];
            double sum = 0.0;
            for (std::size_t l = 0; l < lb; ++l) {
              target[cb[l]] += d*source[l];                            // R_P[., loc(B)] += D[A,B] W_P
              sum += row[cb[l]]*source[l];
            }
            target[offset[b]+nu] += sum;                               // R'_P[., B] += D[A,loc(B)] W_P^T
          }
        }

      // Gamma_PQ = Tr[D C^P D C^Q], three sums.
      // H1 [P, j, l] = sum_i D[B_j, A_i] G_P[i, loc(B)_l]
      h1.assign(np*ub*lb,0.0);
      for (std::size_t p = 0; p < np; ++p)
        for (std::size_t j = 0; j < ub; ++j)
          for (std::size_t i = 0; i < ua; ++i) {
            const double d = density[(offset[b]+j)*n+offset[a]+i];
            const double* source = g[a].data()+(p*ua+i)*n;
            double* target = h1.data()+(p*ub+j)*lb;
            for (std::size_t l = 0; l < lb; ++l) target[l] += d*source[cb[l]];
          }
      // H2 [Q, i, l] = sum_j D[A_i, B_j] G_Q[j, loc(A)_l]
      h2.assign(nq*ua*la,0.0);
      for (std::size_t q = 0; q < nq; ++q)
        for (std::size_t i = 0; i < ua; ++i)
          for (std::size_t j = 0; j < ub; ++j) {
            const double d = density[(offset[a]+i)*n+offset[b]+j];
            const double* source = g[b].data()+(q*ub+j)*n;
            double* target = h2.data()+(q*ua+i)*la;
            for (std::size_t l = 0; l < la; ++l) target[l] += d*source[ca[l]];
          }
      std::vector<double>& dv = gradient.v[a][ib];
      dv.assign(np*nq,0.0);
      for (std::size_t p = 0; p < np; ++p)
        for (std::size_t q = 0; q < nq; ++q) {
          double sum = 0.0;
          for (std::size_t e = 0; e < ub*lb; ++e) sum += tb[q*ub*lb+e]*h1[p*ub*lb+e];
          for (std::size_t e = 0; e < ua*la; ++e) sum += ta[p*ua*la+e]*h2[q*ua*la+e];
          for (std::size_t i = 0; i < ua; ++i)
            for (std::size_t j = 0; j < ub; ++j)
              sum += 2.0*g[a][(p*ua+i)*n+offset[b]+j]*g[b][(q*ub+j)*n+offset[a]+i];
          dv[p*nq+q] = -0.5*alpha*sum;
        }
    }

    // F^P = (R_P + R'_P) D[., loc(A)], and dE/dT = -2 alpha F^P.
    columns_of_density.resize(n*la);
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t l = 0; l < la; ++l) columns_of_density[row*la+l] = density[row*n+folded[a][l]];
    gradient.t[a].assign(rows*la,0.0);
    backend.multiply(rows,la,n,-2.0*alpha,bracket.data(),columns_of_density.data(),0.0,gradient.t[a].data());
  }
  return gradient;
}

} // namespace openmx::exx
