#include <openmx/exx/exchange.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <type_traits>
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

// The transpose of the rows x columns array of block, written over it.
void transpose_in_place(double* block, std::size_t rows, std::size_t columns, std::vector<double>& scratch)
{
  scratch.assign(block,block+rows*columns);
  for (std::size_t c = 0; c < columns; ++c)
    for (std::size_t r = 0; r < rows; ++r) block[c*rows+r] = scratch[r*columns+c];
}

// A run of neighbouring columns in a list of columns: the entries first to
// first+count-1 of the list are the columns column to column+count-1.
struct Run {
  std::size_t first, column, count;
};

void find_runs(const std::vector<std::size_t>& column, std::vector<Run>& runs)
{
  runs.clear();
  for (std::size_t l = 0; l < column.size();) {
    std::size_t count = 1;
    while (l+count < column.size() && column[l+count] == column[l]+count) ++count;
    runs.push_back(Run{l,column[l],count});
    l += count;
  }
}

// The arrays of one piece of the ABFs of a centre atom: the pieces of
// centre_contribution() can run at the same time.
struct PieceWork {
  std::vector<double> scratch, gathered, x3, x4, k1, second;
};

// The number of pieces of the work of `count` ABFs of a centre atom: the
// threads of the backend, or the ABFs when they are fewer.  Every loop of the
// build of these ABFs is asked for one call for each piece.
std::size_t piece_count(std::size_t count, const DenseBackend& backend)
{
  return std::max<std::size_t>(1,std::min(backend.pieces(),count));
}

// The first ABF of every atom in the count over the ABFs of all atoms, and
// their number at the end.
std::vector<std::size_t> abf_offsets(const System& system)
{
  std::vector<std::size_t> offset(1,0);
  for (const std::size_t n : system.abf_count) offset.push_back(offset.back()+n);
  return offset;
}

// The ABFs first <= P < last of one centre atom, counted within the atom.
struct CentrePart {
  std::size_t atom, first, last;
};

// The parts of the centre atoms that the ABFs first <= P < last hold, P in
// the count over all atoms (abf_offsets()), in the order of the atoms.
std::vector<CentrePart> centre_parts(const std::vector<std::size_t>& offset, std::size_t first, std::size_t last)
{
  std::vector<CentrePart> parts;
  for (std::size_t a = 0; a+1 < offset.size(); ++a) {
    const std::size_t from = std::max(first,offset[a]), to = std::min(last,offset[a+1]);
    if (from < to) parts.push_back(CentrePart{a,from-offset[a],to-offset[a]});
  }
  return parts;
}

// The contribution of the centre atom a, section 6.4.  The columns of the
// density matrix are numbered 0..s-1 by the caller: the orbitals of the sites
// of S in the site form, the orbitals of the unit cell in the Gamma form.
// dloc is D[loc(A), columns], width(a) x s; its first u_a rows belong to A.
//
// The contribution is a sum over the ABFs P of the centre: G, W, the four
// terms and the last product are separate for every P.  The call computes
// the part of the ABFs p_first <= P < p_last, and the parts of an atom add up
// to its contribution (section 6.7).  Below, np is the number of ABFs of the
// part, and P counts them from 0.
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
//   last(t4, backend, pieces)
//                   rows loc(A), all columns      terms (iii) and (iv)
// Term (iii) is added to the bracket of term (iv) at the columns of B, so one
// product with the T blocks transforms both.  A sink may add the last result
// in a loop of the backend with one call for each piece.
//
// Layouts, with P and mu on A, Q and nu on B, l a column of loc(B):
//   T_a [(P,mu), lambda]   W [(P,nu), l]   G and the bracket [(P,column), mu]
// The two products with the T blocks of the centre give and take G and the
// bracket as [(P,mu), column]; the arrays are turned in between.  The part of
// one P is then one piece of memory, and the entries of one column lie
// together in it.
//
// The four terms of a V neighbour are computed for one P at a time.  W_P is a
// matrix [nu, l] that a product takes as it is or transposed.  One product
// over all P needs a second copy of W with l before nu and the columns of
// loc(B) of all of G in a new order; that form took 1.5 to 1.8 times as long
// (section 12 of the implementation guide).
//
// The work of the single P runs in pieces: the loop of the backend may give
// every piece to another thread.  Piece c holds the ABFs c np/pieces to
// (c+1) np/pieces - 1, with integer divisions.  The parts of G and of the
// bracket of different P are different memory.  Terms (i) and (ii) are sums
// over P: every piece adds up its own P in arrays of its own, and the sums of
// the pieces are added in the order of the pieces.  With one piece the
// operations and their order are those of a loop over all P.  The number of
// pieces is piece_count(); the callers that gather dloc in pieces use the
// same number.
template <class Sink>
void centre_contribution(const System& system, std::size_t a, std::size_t p_first, std::size_t p_last, std::size_t s,
                         const double* dloc, Sink& sink, const DenseBackend& backend)
{
  const std::size_t np = p_last-p_first, ua = system.orbital_count[a], ell = sink.width(a);
  const std::size_t rows = np*ua, part = s*ua;
  const double* tst = sink.t(a)+p_first*ua*ell;                        // the rows of the ABFs of the part
  const double* da = dloc;

  const std::size_t pieces = piece_count(np,backend);
  const auto begin = [np,pieces](std::size_t c) { return c*np/pieces; };
  std::vector<PieceWork> work(pieces);

  // The products write every element of G and of the last result: the two
  // arrays are not set to zero first.  The bracket is set to zero by the
  // pieces that fill it.
  const std::unique_ptr<double[]> g(new double[rows*s]), bracket(new double[rows*s]);
  backend.multiply(rows,s,ell,1.0,tst,dloc,0.0,g.get());               // G = T D[loc(A), .]
  backend.parallel_for(pieces,[&](std::size_t c) {
    for (std::size_t p = begin(c); p < begin(c+1); ++p) {
      transpose_in_place(g.get()+p*part,ua,s,work[c].scratch);
      std::fill_n(bracket.get()+p*part,part,0.0);
    }
  });

  std::vector<Run> runs;
  std::vector<double> w, dl, dba, first;
  for (std::size_t ib = 0; ib < sink.groups(); ++ib) {
    const std::size_t b = sink.atom(ib);
    const std::size_t nq = system.abf_count[b], ub = system.orbital_count[b], lb = sink.width(b);
    const std::vector<std::size_t>& column = sink.columns(ib);         // lb entries
    find_runs(column,runs);

    // W_{P,B} = sum_Q V_PQ T_Q, equation (10), for the rows P of the part
    w.resize(np*ub*lb);
    backend.multiply(np,ub*lb,nq,1.0,sink.v(ib)+p_first*nq,sink.t(b),0.0,w.data());

    // D[A, loc(B)] as [mu, l] and D[A, B] as [nu, mu]
    dl.resize(ua*lb);
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t l = 0; l < lb; ++l) dl[mu*lb+l] = da[mu*s+column[l]];
    dba.resize(ub*ua);
    for (std::size_t nu = 0; nu < ub; ++nu)
      for (std::size_t mu = 0; mu < ua; ++mu) dba[nu*ua+mu] = da[mu*s+column[nu]];

    backend.parallel_for(pieces,[&](std::size_t c) {
      PieceWork& mine = work[c];
      mine.gathered.resize(lb*ua);
      mine.x3.resize(ub*ua);
      mine.x4.resize(lb*ua);
      mine.k1.assign(ub*ua,0.0);                                         // term (i) as [nu, mu]
      mine.second.assign(ua*lb,0.0);
      for (std::size_t p = begin(c); p < begin(c+1); ++p) {
        const double* wp = w.data()+p*ub*lb;                             // W_P [nu, l]
        const double* gp = g.get()+p*part;                               // G_P [column, mu]
        double* bp = bracket.get()+p*part;                               // bracket_P [column, mu]

        // G_P[loc(B), .] as [l, mu]; its first u_b rows are G_P[B, .]
        for (const Run& run : runs) std::copy_n(gp+run.column*ua,run.count*ua,mine.gathered.data()+run.first*ua);

        // (i)  sum_l W_P[., l] G_P[loc(B), .]
        backend.multiply(ub,ua,lb,1.0,wp,mine.gathered.data(),1.0,mine.k1.data());

        // (ii)  sum_nu G_P[B, .]^T W_P
        backend.transposed_multiply(ua,lb,ub,1.0,mine.gathered.data(),wp,1.0,mine.second.data());

        // (iii)  bracket_P[B, .] += W_P D[A, loc(B)]^T
        backend.multiply_transposed(ub,ua,lb,1.0,wp,dl.data(),0.0,mine.x3.data());
        for (std::size_t nu = 0; nu < ub; ++nu) {
          double* target = bp+column[nu]*ua;
          for (std::size_t mu = 0; mu < ua; ++mu) target[mu] += mine.x3[nu*ua+mu];
        }

        // (iv)  bracket_P[loc(B), .] += W_P^T D[A, B]^T
        backend.transposed_multiply(lb,ua,ub,1.0,wp,dba.data(),0.0,mine.x4.data());
        for (const Run& run : runs) {
          double* target = bp+run.column*ua;
          const double* source = mine.x4.data()+run.first*ua;
          for (std::size_t e = 0; e < run.count*ua; ++e) target[e] += source[e];
        }
      }
    });

    // The sums of the pieces, added to those of the first piece in the order
    // of the pieces.  The second array is long: its elements are divided
    // among the calls of a loop, and every element is added up by one call.
    std::vector<double>& k1 = work[0].k1;
    std::vector<double>& second = work[0].second;
    for (std::size_t c = 1; c < pieces; ++c)
      for (std::size_t e = 0; e < ub*ua; ++e) k1[e] += work[c].k1[e];
    if (pieces > 1)
      backend.parallel_for(pieces,[&](std::size_t share) {
        const std::size_t from = share*(ua*lb)/pieces, to = (share+1)*(ua*lb)/pieces;
        for (std::size_t c = 1; c < pieces; ++c) {
          const double* source = work[c].second.data();
          for (std::size_t e = from; e < to; ++e) second[e] += source[e];
        }
      });

    first.resize(ua*ub);
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t nu = 0; nu < ub; ++nu) first[mu*ub+nu] = k1[nu*ua+mu];
    sink.first(ib,first.data());
    sink.second(ib,second.data());
  }

  // rows loc(A):  sum_P T_P^T bracket_P
  backend.parallel_for(pieces,[&](std::size_t c) {
    for (std::size_t p = begin(c); p < begin(c+1); ++p)
      transpose_in_place(bracket.get()+p*part,s,ua,work[c].scratch);
  });
  const std::unique_ptr<double[]> last(new double[ell*s]);
  backend.transposed_multiply(ell,s,rows,1.0,tst,bracket.get(),0.0,last.get());
  sink.last(last.get(),backend,pieces);
}

// The derivatives of E_x = (1/2) sum D K with respect to the blocks of the
// centre atom a, for K = factor times the sum of the contributions of all
// centres and a symmetric density matrix: equations (16) and (17) with the
// arrays of centre_contribution().  The sink is used for its blocks and
// columns only.
//
// The contribution of the centre to E_x is factor times the sum over P, mu
// and the columns of G_P times the bracket of terms (iii) and (iv).
//   dt [(P,mu), lambda] = 2 factor sum_column bracket_P[column, mu] D[loc(A)_lambda, column]
// The T blocks of the centre enter this sum through G and the sums of other
// centres through W; the two parts are equal, hence the factor 2.
//   dv[g] [P, Q] = factor sum_{nu l} T_Q[nu, l] H_P[nu, l],
//   H_P = D[B, A] G_P[., loc(B)] + G_P[B, .] D[A, loc(B)]
// The V block of group g enters the sum of this centre only.
//
// Both derivatives are separate for every ABF P of the centre, and the call
// computes those of the ABFs p_first <= P < p_last: dt and dv[g] get the
// rows of these ABFs, np = p_last - p_first of them, and nothing else.
template <class Sink>
void centre_derivatives(const System& system, std::size_t a, std::size_t p_first, std::size_t p_last, std::size_t s,
                        const std::vector<double>& dloc, Sink& sink, double factor, const DenseBackend& backend,
                        std::vector<double>& dt, std::vector<std::vector<double>>& dv)
{
  const std::size_t np = p_last-p_first, ua = system.orbital_count[a], ell = sink.width(a);
  const std::size_t rows = np*ua, part = s*ua;
  const double* tst = sink.t(a)+p_first*ua*ell;
  const double* da = dloc.data();

  const std::unique_ptr<double[]> g(new double[rows*s]);
  std::vector<double> bracket(rows*s,0.0), scratch;
  backend.multiply(rows,s,ell,1.0,tst,dloc.data(),0.0,g.get());       // G = T D[loc(A), .]
  for (std::size_t p = 0; p < np; ++p) transpose_in_place(g.get()+p*part,ua,s,scratch);

  std::vector<Run> runs;
  std::vector<double> w, dl, dba, gathered, h, x3, x4;
  dv.resize(sink.groups());
  for (std::size_t ib = 0; ib < sink.groups(); ++ib) {
    const std::size_t b = sink.atom(ib);
    const std::size_t nq = system.abf_count[b], ub = system.orbital_count[b], lb = sink.width(b);
    const std::vector<std::size_t>& column = sink.columns(ib);         // lb entries
    find_runs(column,runs);

    w.resize(np*ub*lb);                                                // W_{P,B} = sum_Q V_PQ T_Q
    backend.multiply(np,ub*lb,nq,1.0,sink.v(ib)+p_first*nq,sink.t(b),0.0,w.data());

    dl.resize(ua*lb);                                                  // D[A, loc(B)] as [mu, l]
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t l = 0; l < lb; ++l) dl[mu*lb+l] = da[mu*s+column[l]];
    dba.resize(ub*ua);                                                 // D[A, B] as [nu, mu]
    for (std::size_t nu = 0; nu < ub; ++nu)
      for (std::size_t mu = 0; mu < ua; ++mu) dba[nu*ua+mu] = da[mu*s+column[nu]];

    gathered.resize(lb*ua);
    x3.resize(ub*ua);
    x4.resize(lb*ua);
    h.resize(np*ub*lb);
    for (std::size_t p = 0; p < np; ++p) {
      const double* wp = w.data()+p*ub*lb;                             // W_P [nu, l]
      const double* gp = g.get()+p*part;                               // G_P [column, mu]
      double* bp = bracket.data()+p*part;                              // bracket_P [column, mu]
      double* hp = h.data()+p*ub*lb;                                   // H_P [nu, l]

      // G_P[loc(B), .] as [l, mu]; its first u_b rows are G_P[B, .]
      for (const Run& run : runs) std::copy_n(gp+run.column*ua,run.count*ua,gathered.data()+run.first*ua);

      // (iii)  bracket_P[B, .] += W_P D[A, loc(B)]^T
      backend.multiply_transposed(ub,ua,lb,1.0,wp,dl.data(),0.0,x3.data());
      for (std::size_t nu = 0; nu < ub; ++nu) {
        double* target = bp+column[nu]*ua;
        for (std::size_t mu = 0; mu < ua; ++mu) target[mu] += x3[nu*ua+mu];
      }

      // (iv)  bracket_P[loc(B), .] += W_P^T D[A, B]^T
      backend.transposed_multiply(lb,ua,ub,1.0,wp,dba.data(),0.0,x4.data());
      for (const Run& run : runs) {
        double* target = bp+run.column*ua;
        const double* source = x4.data()+run.first*ua;
        for (std::size_t e = 0; e < run.count*ua; ++e) target[e] += source[e];
      }

      // H_P = D[B, A] G_P[., loc(B)] + G_P[B, .] D[A, loc(B)]
      backend.multiply_transposed(ub,lb,ua,1.0,dba.data(),gathered.data(),0.0,hp);
      backend.multiply(ub,lb,ua,1.0,gathered.data(),dl.data(),1.0,hp);
    }
    dv[ib].resize(np*nq);
    backend.multiply_transposed(np,nq,ub*lb,factor,h.data(),sink.t(b),0.0,dv[ib].data());
  }

  for (std::size_t p = 0; p < np; ++p) transpose_in_place(bracket.data()+p*part,s,ua,scratch);
  dt.resize(rows*ell);
  backend.multiply_transposed(rows,ell,s,2.0*factor,bracket.data(),dloc.data(),0.0,dt.data());
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

std::array<int,4> site_key(const Site& site)
{
  return {site.atom,site.cell[0],site.cell[1],site.cell[2]};
}

// S(A) of the centre atom a: the sites of loc(B) for all V neighbours B, in
// the order of their first appearance.
std::vector<Site> column_sites(const System& system, std::size_t a)
{
  std::vector<Site> sites;
  std::set<std::array<int,4>> seen;
  for (const Site& b : system.v_neighbours[a])
    for (const Site& c : system.c_neighbours[static_cast<std::size_t>(b.atom)]) {
      const Site site{c.atom,sum(b.cell,c.cell)};
      if (seen.insert(site_key(site)).second) sites.push_back(site);
    }
  return sites;
}

class LayoutBlocks;

// Blocks is the store of the blocks of K: at(key) gives the block of a key,
// u_row x u_column elements.  With the store of an array in a layout
// (LayoutBlocks) the sink also gathers D and adds the last result in pieces.
template <class Blocks>
class SiteSink {
 public:
  SiteSink(const System& system, std::size_t a, double factor, Blocks& k)
      : system_(system), a_(a), factor_(factor), k_(k), sites_(column_sites(system,a)), offset_(1,0)
  {
    for (const Site& site : sites_) {
      index_.emplace(site_key(site),offset_.size()-1);
      offset_.push_back(offset_.back()+system.orbital_count[static_cast<std::size_t>(site.atom)]);
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

  // D[loc(A), S] from an array of D in the layout of the store of K.  The
  // sites of loc(A) are divided into pieces, and the pieces copy the rows of
  // their sites; a loop of the backend may run them at the same time.  The
  // blocks of the pairs of sites cover the array, so every element is
  // written.  The index of the block of every pair is kept for last().
  std::unique_ptr<double[]> density(std::span<const double> values, const DenseBackend& backend, std::size_t pieces)
  {
    static_assert(std::is_same_v<Blocks,LayoutBlocks>,"only a store in a layout gathers D in pieces");
    const BlockLayout& layout = k_.layout();
    const std::vector<Site>& loc = system_.c_neighbours[a_];
    const std::vector<std::size_t>& first = system_.c_offset[a_];      // the first row of every site of loc(A)
    const std::size_t s = size(), count = sites_.size();
    std::unique_ptr<double[]> dloc(new double[first.back()*s]);
    block_.resize(loc.size()*count);
    backend.parallel_for(pieces,[&](std::size_t c) {
      for (std::size_t i = c*loc.size()/pieces; i < (c+1)*loc.size()/pieces; ++i) {
        const std::size_t rows = first[i+1]-first[i];
        for (std::size_t j = 0; j < count; ++j) {
          const std::size_t b = layout.find(BlockKey{loc[i].atom,sites_[j].atom,difference(sites_[j].cell,loc[i].cell)});
          if (b == layout.keys.size()) throw std::invalid_argument("the layout lacks a block of the density matrix");
          block_[i*count+j] = b;
          const std::size_t columns = offset_[j+1]-offset_[j];
          const double* source = values.data()+layout.offset[b];
          for (std::size_t r = 0; r < rows; ++r) std::copy_n(source+r*columns,columns,dloc.get()+(first[i]+r)*s+offset_[j]);
        }
      }
    });
    return dloc;
  }

  // For the device (section 11): every row of every pair of a site i of
  // loc(A) and a site j of S, in the order of density(), with its offset in
  // D[loc(A), S] (local), the offset of the same row of the block in the
  // layout (stored) and its length.  The rows of site i begin at begin[i].
  // Two rows of one site of loc(A) never lie in the same block of the
  // layout, so the rows of one site can be added to K at the same time.
  void device_rows(std::vector<long long>& local, std::vector<long long>& stored, std::vector<int>& length,
                   std::vector<std::size_t>& begin) const
  {
    static_assert(std::is_same_v<Blocks,LayoutBlocks>,"only a store in a layout gives the rows for the device");
    const BlockLayout& layout = k_.layout();
    const std::vector<Site>& loc = system_.c_neighbours[a_];
    const std::vector<std::size_t>& first = system_.c_offset[a_];
    const std::size_t s = size();
    begin.assign(1,local.size());
    for (std::size_t i = 0; i < loc.size(); ++i) {
      for (std::size_t j = 0; j < sites_.size(); ++j) {
        const std::size_t b = layout.find(BlockKey{loc[i].atom,sites_[j].atom,difference(sites_[j].cell,loc[i].cell)});
        if (b == layout.keys.size()) throw std::invalid_argument("the layout lacks a block of the density matrix");
        const std::size_t columns = offset_[j+1]-offset_[j];
        for (std::size_t r = first[i]; r < first[i+1]; ++r) {
          local.push_back(static_cast<long long>(r*s+offset_[j]));
          stored.push_back(static_cast<long long>(layout.offset[b]+(r-first[i])*columns));
          length.push_back(static_cast<int>(columns));
        }
      }
      begin.push_back(local.size());
    }
  }

  // The offset in the layout of the block of K of term (i) of group ib.
  long long device_first(std::size_t ib) const
  {
    const Site& b = system_.v_neighbours[a_][ib];
    return block_offset(BlockKey{static_cast<int>(a_),b.atom,b.cell});
  }

  // The rows of term (ii) of group ib: for every site of loc(B) and every
  // orbital mu of A the offset of the row in the layout (stored), the offset
  // in the array of the term, u_a x l_B (local), and the length.  The sites
  // of loc(B) are different blocks.
  void device_second(std::size_t ib, std::vector<long long>& stored, std::vector<long long>& local,
                     std::vector<int>& length) const
  {
    const Site& b = system_.v_neighbours[a_][ib];
    const std::size_t atom = static_cast<std::size_t>(b.atom);
    const std::size_t ua = system_.orbital_count[a_], lb = system_.c_offset[atom].back();
    const std::vector<Site>& loc = system_.c_neighbours[atom];
    for (std::size_t j = 0; j < loc.size(); ++j) {
      const std::size_t begin = system_.c_offset[atom][j], width = system_.c_offset[atom][j+1]-begin;
      const long long block = block_offset(BlockKey{static_cast<int>(a_),loc[j].atom,sum(b.cell,loc[j].cell)});
      for (std::size_t mu = 0; mu < ua; ++mu) {
        stored.push_back(block+static_cast<long long>(mu*width));
        local.push_back(static_cast<long long>(mu*lb+begin));
        length.push_back(static_cast<int>(width));
      }
    }
  }

  // The number of sites of S.
  std::size_t site_count() const noexcept { return sites_.size(); }

  const std::vector<std::size_t>& columns(std::size_t ib)
  {
    const Site& b = system_.v_neighbours[a_][ib];
    columns_.clear();
    for (const Site& c : system_.c_neighbours[static_cast<std::size_t>(b.atom)]) {
      const std::size_t j = index_.at(site_key(Site{c.atom,sum(b.cell,c.cell)}));
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

  // K[loc(A), S] += factor values.  The store of a layout takes the blocks
  // that density() found, in pieces of the blocks: piece c adds to the
  // blocks whose index in the layout leaves the remainder c when divided by
  // the number of pieces.  Two pairs of sites share a block of K when loc(A)
  // holds two images of one atom.  Every piece goes through the pairs in the
  // order of the loop on one thread, so every element receives its additions
  // in the same order.  Another store, such as the map of a BlockMatrix,
  // takes the blocks on the calling thread.
  void last(const double* values, const DenseBackend& backend, std::size_t pieces)
  {
    const std::vector<Site>& loc = system_.c_neighbours[a_];
    const std::size_t s = size(), count = sites_.size();
    if constexpr (std::is_same_v<Blocks,LayoutBlocks>) {
      if (block_.size() != loc.size()*count)
        throw std::logic_error("the blocks of the exchange matrix are added before they are found");
      backend.parallel_for(pieces,[&](std::size_t piece) {
        for (std::size_t i = 0; i < loc.size(); ++i) {
          const std::size_t begin = system_.c_offset[a_][i], rows = system_.c_offset[a_][i+1]-begin;
          for (std::size_t j = 0; j < count; ++j) {
            const std::size_t b = block_[i*count+j];
            if (b%pieces != piece) continue;
            const std::size_t columns = offset_[j+1]-offset_[j];
            const std::span<double> block = k_.block(b);
            for (std::size_t r = 0; r < rows; ++r)
              for (std::size_t c = 0; c < columns; ++c) block[r*columns+c] += factor_*values[(begin+r)*s+offset_[j]+c];
          }
        }
      });
    } else {
      (void)backend;
      (void)pieces;
      for (std::size_t i = 0; i < loc.size(); ++i) {
        const std::size_t begin = system_.c_offset[a_][i], rows = system_.c_offset[a_][i+1]-begin;
        for (std::size_t j = 0; j < count; ++j) {
          const std::size_t columns = offset_[j+1]-offset_[j];
          const std::span<double> block = k_.at(BlockKey{loc[i].atom,sites_[j].atom,difference(sites_[j].cell,loc[i].cell)});
          for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < columns; ++c) block[r*columns+c] += factor_*values[(begin+r)*s+offset_[j]+c];
        }
      }
    }
  }

 private:
  long long block_offset(const BlockKey& key) const
  {
    static_assert(std::is_same_v<Blocks,LayoutBlocks>,"only a store in a layout gives offsets for the device");
    const BlockLayout& layout = k_.layout();
    const std::size_t i = layout.find(key);
    if (i == layout.keys.size()) throw std::invalid_argument("the layout lacks a block of the exchange matrix");
    return static_cast<long long>(layout.offset[i]);
  }

  const System& system_;
  std::size_t a_;
  double factor_;
  Blocks& k_;
  std::vector<Site> sites_;                         // S
  std::vector<std::size_t> offset_;                 // first column of every site of S
  std::map<std::array<int,4>,std::size_t> index_;   // position of a site in S
  std::vector<std::size_t> columns_;
  std::vector<std::size_t> block_;                  // the block in the layout of every pair (i, j) of a site of
                                                    // loc(A) and a site of S, at i |S| + j; from density()
};

// ---- site form in one array --------------------------------------------------

// A layout must fit the system: the atoms of its keys exist, and every block
// has the size that the orbitals of its two atoms give.
void check(const System& system, const BlockLayout& layout)
{
  const std::size_t atoms = system.orbital_count.size(), blocks = layout.keys.size();
  if (layout.offset.size() != blocks+1 || layout.offset.front() != 0)
    throw std::invalid_argument("the offsets of a layout do not match its keys");
  for (std::size_t i = 0; i < blocks; ++i) {
    const BlockKey& key = layout.keys[i];
    if (key.row < 0 || key.column < 0 || static_cast<std::size_t>(key.row) >= atoms ||
        static_cast<std::size_t>(key.column) >= atoms)
      throw std::invalid_argument("an atom of a block of the layout does not exist");
    const std::size_t size = system.orbital_count[static_cast<std::size_t>(key.row)]*
                             system.orbital_count[static_cast<std::size_t>(key.column)];
    if (layout.offset[i+1] < layout.offset[i] || layout.offset[i+1]-layout.offset[i] != size)
      throw std::invalid_argument("a block of the layout does not match the orbitals of its atoms");
  }
}

// The blocks of an array in a layout.
class LayoutBlocks {
 public:
  LayoutBlocks(const BlockLayout& layout, std::span<double> values) : layout_(layout), values_(values) {}

  std::span<double> at(const BlockKey& key) const
  {
    const std::size_t i = layout_.find(key);
    if (i == layout_.keys.size()) throw std::invalid_argument("the layout lacks a block of the exchange matrix");
    return block(i);
  }

  // The block of index i in the layout.
  std::span<double> block(std::size_t i) const
  {
    return values_.subspan(layout_.offset[i],layout_.offset[i+1]-layout_.offset[i]);
  }

  const BlockLayout& layout() const noexcept { return layout_; }

 private:
  const BlockLayout& layout_;
  std::span<double> values_;
};

// Calls visit(e, f) once for every two elements e and f of an array in the
// layout that are equal in a matrix with M_xy(R) = M_yx(-R)^T.
template <class Visit>
void for_mirrored(const System& system, const BlockLayout& layout, Visit visit)
{
  check(system,layout);
  for (std::size_t i = 0; i < layout.keys.size(); ++i) {
    const BlockKey& key = layout.keys[i];
    const std::size_t j = layout.find(BlockKey{key.column,key.row,{-key.cell[0],-key.cell[1],-key.cell[2]}});
    if (j == layout.keys.size()) throw std::invalid_argument("the layout lacks the mirror image of a block");
    if (j < i) continue;                               // visited from the other block
    const std::size_t rows = system.orbital_count[static_cast<std::size_t>(key.row)];
    const std::size_t columns = system.orbital_count[static_cast<std::size_t>(key.column)];
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = (i == j ? r+1 : 0); c < columns; ++c)
        visit(layout.offset[i]+r*columns+c,layout.offset[j]+c*rows+r);
  }
}

// ---- Gamma form ------------------------------------------------------------

// The sites fold onto the atoms of the unit cell: all images of an atom
// address the same orbitals.  The contribution of a centre is linear in the
// V block of a neighbour and in the columns of the T blocks, so the folding
// is done before the products.  The V blocks of the images of an atom are
// added up, and so are the columns of the T blocks that belong to one
// orbital.  A cell that is smaller than the cutoffs then costs the products
// of a V neighbour once for every atom among the V neighbours instead of
// once for every image, and no array has more than N columns.
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

  // The orbitals of loc(A) are different rows of K: the pieces of the rows
  // add to different memory, and every element receives one addition.
  void last(const double* values, const DenseBackend& backend, std::size_t pieces)
  {
    const std::vector<std::size_t>& row = folded(a_).column;
    backend.parallel_for(pieces,[&](std::size_t c) {
      for (std::size_t lambda = c*row.size()/pieces; lambda < (c+1)*row.size()/pieces; ++lambda)
        for (std::size_t column = 0; column < n_; ++column) k_[row[lambda]*n_+column] += factor_*values[lambda*n_+column];
    });
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

// The same rows in pieces, which a loop of the backend may run at the same
// time.  Every row is copied whole, so every element is written.
std::unique_ptr<double[]> gathered_rows(std::span<const double> density, std::size_t n,
                                        const std::vector<std::size_t>& row, const DenseBackend& backend,
                                        std::size_t pieces)
{
  std::unique_ptr<double[]> dloc(new double[row.size()*n]);
  backend.parallel_for(pieces,[&](std::size_t c) {
    for (std::size_t lambda = c*row.size()/pieces; lambda < (c+1)*row.size()/pieces; ++lambda)
      std::copy_n(density.data()+row[lambda]*n,n,dloc.get()+lambda*n);
  });
  return dloc;
}

void check_dense(std::size_t n, std::size_t size, const char* what)
{
  if (size != n*n) throw std::invalid_argument(what);
}

// ---- the device (section 11) ------------------------------------------------
//
// One call of the library builds the contributions of its centre atoms on the
// device.  D, K, the T blocks of all atoms and a zero stay there for the
// call; the arrays of one centre are made for the largest centre and used by
// every centre in turn.  A call can hold a part of the ABFs of a centre
// (CentrePart): the arrays then have the ABFs of the part, as on the host.
// The device has the arrays of centre_contribution() in other layouts, so
// that no array is turned:
//   G and the bracket   [P][mu][column]   (the result of G = T D as it is)
//   W                   [P][nu][l]        (the result of W = V T as it is)
//   Wn                  [nu][P][l]        (a copy of W)
//   Gg                  [mu][P][l] = G[P][mu][column_l]
//   GBp                 [mu][P][nu] = Gg[mu][P][nu], nu < u_b
// and the four terms of a V neighbour are one product each for all ABFs:
//   (i)   first [mu][nu]      = sum_(P,l)  Gg[mu][(P,l)] Wn[nu][(P,l)]
//   (ii)  second [mu][l]      = sum_(P,nu) GBp[mu][(P,nu)] W[(P,nu)][l]
//   (iii) X3 [(nu,P)][mu]     = sum_l Wn[(nu,P)][l] D[A, loc(B)][mu][l]
//   (iv)  X4 [mu][(P,l)]      = sum_nu D[A, B][nu][mu] Wn[nu][(P,l)]
// X3 is added to the bracket at the columns of B and then X4 at the columns
// of loc(B), the order of the host.  K receives term (i), term (ii) and the
// last product in the order of the host as well.

// A function of the device failed or found no room: the call goes back to
// the host.
struct DeviceStop {};

// The work of one call on the device.  The memory goes back when the object
// ends; work that may still run on it is finished first.
class DeviceWork {
 public:
  explicit DeviceWork(const OpenMXExxDevice& device) noexcept : device_(device) {}
  DeviceWork(const DeviceWork&) = delete;
  DeviceWork& operator=(const DeviceWork&) = delete;
  ~DeviceWork()
  {
    if (pending_) (void)device_.finish(device_.context);
    for (void* pointer : blocks_) device_.release(device_.context,pointer);
  }

  template <class T>
  T* allocate(std::size_t count)
  {
    blocks_.reserve(blocks_.size()+1);   // push_back below does not throw
    void* pointer = device_.allocate(device_.context,std::max<std::size_t>(count,1)*sizeof(T));
    if (pointer == nullptr) throw DeviceStop();
    blocks_.push_back(pointer);
    return static_cast<T*>(pointer);
  }

  // The copies are made when no work is pending: the caller finishes first.
  void upload(void* target, const void* source, std::size_t bytes)
  {
    if (bytes > 0 && device_.upload(device_.context,target,source,bytes) != 0) throw DeviceStop();
  }
  void download(void* target, const void* source, std::size_t bytes)
  {
    if (bytes > 0 && device_.download(device_.context,target,source,bytes) != 0) throw DeviceStop();
  }
  // finish waits for the work also when it reports a failure: it is called
  // once.
  void finish()
  {
    pending_ = false;
    if (device_.finish(device_.context) != 0) throw DeviceStop();
  }

  // C = A B, A B^T and A^T B of row-major arrays without padding, as the
  // products of DenseBackend.
  void multiply(std::size_t m, std::size_t n, std::size_t k, const double* a, const double* b, double* c)
  {
    gemm('N','N',n,m,k,b,n,a,k,c,n);
  }
  void multiply_transposed(std::size_t m, std::size_t n, std::size_t k, const double* a, const double* b, double* c)
  {
    gemm('T','N',n,m,k,b,k,a,k,c,n);
  }
  void transposed_multiply(std::size_t m, std::size_t n, std::size_t k, const double* a, const double* b, double* c)
  {
    gemm('N','T',n,m,k,b,n,a,m,c,n);
  }

  // One side of a move: the offset of row r is r stride or rows[r], that of
  // column c is c step or columns[c]; the lists are in device memory.
  struct Side {
    long long stride = 0;
    const long long* rows = nullptr;
    long long step = 1;
    const long long* columns = nullptr;
  };

  // target(r, c) = alpha source(r, c), or += with add, for r < rows and c <
  // length[r] (columns without lengths; columns is the largest length).
  void move(std::size_t rows, std::size_t columns, const int* length, double alpha, bool add, double* target,
            const Side& to, const double* source, const Side& from)
  {
    if (rows == 0 || columns == 0) return;
    if (rows > static_cast<std::size_t>(INT_MAX) || columns > static_cast<std::size_t>(INT_MAX)) throw DeviceStop();
    const OpenMXExxDeviceIndex t{to.stride,to.rows,to.step,to.columns};
    const OpenMXExxDeviceIndex s{from.stride,from.rows,from.step,from.columns};
    pending_ = true;
    if (device_.move(device_.context,static_cast<int>(rows),static_cast<int>(columns),length,alpha,add ? 1 : 0,target,
                     &t,source,&s) != 0)
      throw DeviceStop();
  }

 private:
  void gemm(char transa, char transb, std::size_t m, std::size_t n, std::size_t k, const double* a, std::size_t lda,
            const double* b, std::size_t ldb, double* c, std::size_t ldc)
  {
    if (m == 0 || n == 0) return;
    const std::size_t limit = static_cast<std::size_t>(INT_MAX);
    if (m > limit || n > limit || k > limit || lda > limit || ldb > limit || ldc > limit) throw DeviceStop();
    const auto narrow = [](std::size_t value) { return static_cast<int>(std::max<std::size_t>(value,1)); };
    pending_ = true;
    if (device_.dgemm(device_.context,transa,transb,static_cast<int>(m),static_cast<int>(n),static_cast<int>(k),1.0,
                      a,narrow(lda),b,narrow(ldb),0.0,c,narrow(ldc)) != 0)
      throw DeviceStop();
  }

  const OpenMXExxDevice& device_;
  std::vector<void*> blocks_;
  bool pending_ = false;
};

// A V neighbour of a centre atom on the device: group g of the sink.
struct DeviceGroup {
  std::size_t atom = 0;                    // B
  const double* v = nullptr;               // its V block on the host, n_P x n_Q
  std::vector<long long> column;           // the column of D of every column l of the T blocks of B
  long long first = 0, first_stride = 0;   // term (i): row mu of K at first + mu first_stride
  // Term (ii): rows (site of loc(B), mu) with lists in the site form, or row
  // mu at second + mu second_stride and the columns `column` in the Gamma form.
  std::vector<long long> second_stored, second_local;
  std::vector<int> second_length;
  std::size_t second_longest = 0;          // the largest of second_length
  long long second = 0, second_stride = 0;
};

// What one centre atom needs on the device, from the sink.
struct DeviceCentre {
  std::size_t s = 0;                      // columns of D[loc(A), .]
  // Site form: the rows of D[loc(A), S] with lists (SiteSink::device_rows()).
  std::vector<long long> local, stored;
  std::vector<int> length;
  std::vector<std::size_t> begin;
  std::size_t longest = 0;                // the largest of length
  // Gamma form: the offset in D and K of the row of every row of D[loc(A), .].
  std::vector<long long> row;
  std::vector<DeviceGroup> groups;
};

// The sizes of the arrays of one centre atom on the device.
struct DeviceSizes {
  std::size_t g = 0, dloc = 0, w = 0, gg = 0, gbp = 0, dl = 0, dba = 0, first = 0, second = 0, x3 = 0;
  std::size_t v = 0, list = 0, length = 0;
  double multiplications = 0.0;           // of the largest product of the centre

  void widen(const DeviceSizes& other)
  {
    g = std::max(g,other.g);
    dloc = std::max(dloc,other.dloc);
    w = std::max(w,other.w);
    gg = std::max(gg,other.gg);
    gbp = std::max(gbp,other.gbp);
    dl = std::max(dl,other.dl);
    dba = std::max(dba,other.dba);
    first = std::max(first,other.first);
    second = std::max(second,other.second);
    x3 = std::max(x3,other.x3);
    v = std::max(v,other.v);
    list = std::max(list,other.list);
    length = std::max(length,other.length);
    multiplications = std::max(multiplications,other.multiplications);
  }
};

// The sizes of np ABFs of centre a with the columns s, the width ell of
// loc(A), the groups (atom, width of loc(B), sites of loc(B)) and the rows of
// the gathered D (site form; 0 in the Gamma form).
DeviceSizes device_sizes(const System& system, std::size_t a, std::size_t np, std::size_t s, std::size_t ell,
                         const std::vector<std::array<std::size_t,3>>& groups, std::size_t gathered_rows, bool site)
{
  const std::size_t ua = system.orbital_count[a];
  const auto product = [](std::size_t m, std::size_t n, std::size_t k) {
    return static_cast<double>(m)*static_cast<double>(n)*static_cast<double>(k);
  };
  DeviceSizes z;
  z.g = np*ua*s;
  z.dloc = ell*s;
  z.multiplications = product(np*ua,s,ell);         // G = T D and the last product
  z.list = site ? 2*gathered_rows : ell;
  z.length = site ? gathered_rows : 0;
  z.list += ua*np;                                  // the rows of G in the order (mu, P)
  for (const std::array<std::size_t,3>& group : groups) {
    const std::size_t b = group[0], lb = group[1], sites = group[2];
    const std::size_t nq = system.abf_count[b], ub = system.orbital_count[b];
    // W = V T_B, and the four terms, which have the same size.
    z.multiplications = std::max({z.multiplications,product(np,ub*lb,nq),product(ua,ub,np*lb)});
    z.w = std::max(z.w,np*ub*lb);
    z.gg = std::max(z.gg,ua*np*lb);
    z.gbp = std::max(z.gbp,ua*np*ub);
    z.dl = std::max(z.dl,ua*lb);
    z.dba = std::max(z.dba,ub*ua);
    z.first = std::max(z.first,ua*ub);
    z.second = std::max(z.second,ua*lb);
    z.x3 = std::max(z.x3,ub*np*ua);
    z.v += np*nq;
    z.list += lb+ub*np+(site ? 2*sites*ua : 0);
    z.length += site ? sites*ua : 0;
  }
  return z;
}

// The arrays of a call on the device.
struct DeviceArrays {
  double *density = nullptr, *k = nullptr, *t = nullptr, *zero = nullptr, *v = nullptr;
  double *g = nullptr, *bracket = nullptr, *dloc = nullptr, *w = nullptr, *wn = nullptr, *gg = nullptr, *x4 = nullptr;
  double *gbp = nullptr, *dl = nullptr, *dba = nullptr, *first = nullptr, *second = nullptr, *x3 = nullptr;
  long long* list = nullptr;
  int* length = nullptr;
  std::vector<std::size_t> t_offset;      // of the T blocks of every atom in t
  DeviceSizes capacity;                   // of the arrays of one centre

  DeviceArrays(DeviceWork& work, const DeviceSizes& z, std::size_t elements, const std::vector<std::size_t>& t_size)
      : capacity(z)
  {
    t_offset.assign(1,0);
    for (const std::size_t size : t_size) t_offset.push_back(t_offset.back()+size);
    density = work.allocate<double>(elements);
    k = work.allocate<double>(elements);
    t = work.allocate<double>(t_offset.back());
    zero = work.allocate<double>(1);
    v = work.allocate<double>(z.v);
    g = work.allocate<double>(z.g);
    bracket = work.allocate<double>(z.g);
    dloc = work.allocate<double>(z.dloc);
    w = work.allocate<double>(z.w);
    wn = work.allocate<double>(z.w);
    gg = work.allocate<double>(z.gg);
    x4 = work.allocate<double>(z.gg);
    gbp = work.allocate<double>(z.gbp);
    dl = work.allocate<double>(z.dl);
    dba = work.allocate<double>(z.dba);
    first = work.allocate<double>(z.first);
    second = work.allocate<double>(z.second);
    x3 = work.allocate<double>(z.x3);
    list = work.allocate<long long>(z.list);
    length = work.allocate<int>(z.length);
  }

  // The device memory of these arrays, in bytes, as the allocations count it.
  static std::size_t bytes(const DeviceSizes& z, std::size_t elements, std::size_t t_total)
  {
    const auto one = [](std::size_t count) { return std::max<std::size_t>(count,1); };
    const std::size_t doubles = 2*one(elements)+one(t_total)+1+one(z.v)+2*one(z.g)+one(z.dloc)+2*one(z.w)+
                                2*one(z.gg)+one(z.gbp)+one(z.dl)+one(z.dba)+one(z.first)+one(z.second)+one(z.x3);
    return doubles*sizeof(double)+one(z.list)*sizeof(long long)+one(z.length)*sizeof(int);
  }
};

// The contribution of the ABFs of a part of one centre atom on the device,
// factor times it added to K there.  The lists and the V blocks of the
// centre are uploaded first: the work of the centre before has finished.
// The work of this centre is finished at the end.
void device_centre(const System& system, const CentrePart& share, double factor, const DeviceCentre& c,
                   DeviceArrays& d, DeviceWork& work)
{
  const std::size_t a = share.atom, np = share.last-share.first, ua = system.orbital_count[a], s = c.s;
  const std::size_t rows = np*ua;
  const bool site = c.row.empty();
  const std::size_t ell = site ? system.c_offset[a].back() : c.row.size();

  // The lists, in one array: append() gives the offset of a list in it.
  std::vector<long long> list;
  std::vector<int> length;
  const auto append = [&list](const std::vector<long long>& values) {
    const std::size_t at = list.size();
    list.insert(list.end(),values.begin(),values.end());
    return at;
  };
  const auto append_length = [&length](const std::vector<int>& values) {
    const std::size_t at = length.size();
    length.insert(length.end(),values.begin(),values.end());
    return at;
  };
  std::size_t local = 0, stored = 0, gathered = 0, row = 0;
  if (site) {
    local = append(c.local);
    stored = append(c.stored);
    gathered = append_length(c.length);
  } else {
    row = append(c.row);
  }
  std::vector<long long> order(ua*np);             // the row of G of every row (mu, P) of Gg
  for (std::size_t mu = 0; mu < ua; ++mu)
    for (std::size_t p = 0; p < np; ++p) order[mu*np+p] = static_cast<long long>((p*ua+mu)*s);
  const std::size_t g_rows = append(order);
  struct GroupLists {
    std::size_t column, w_rows, second_stored, second_local, second_length;
  };
  std::vector<GroupLists> at(c.groups.size());
  std::vector<double> v;
  std::vector<std::size_t> v_offset;
  for (std::size_t ib = 0; ib < c.groups.size(); ++ib) {
    const DeviceGroup& group = c.groups[ib];
    const std::size_t b = group.atom, ub = system.orbital_count[b], lb = group.column.size();
    at[ib].column = append(group.column);
    std::vector<long long> w_rows(ub*np);            // the row of W of every row (nu, P) of Wn
    for (std::size_t nu = 0; nu < ub; ++nu)
      for (std::size_t p = 0; p < np; ++p) w_rows[nu*np+p] = static_cast<long long>((p*ub+nu)*lb);
    at[ib].w_rows = append(w_rows);
    at[ib].second_stored = append(group.second_stored);
    at[ib].second_local = append(group.second_local);
    at[ib].second_length = append_length(group.second_length);
    v_offset.push_back(v.size());
    const std::size_t nq = system.abf_count[b];
    v.insert(v.end(),group.v+share.first*nq,group.v+share.last*nq);       // the rows of the ABFs of the part
  }
  if (list.size() > d.capacity.list || length.size() > d.capacity.length || v.size() > d.capacity.v)
    throw std::logic_error("the lists of a centre atom do not fit their arrays on the device");
  work.upload(d.list,list.data(),list.size()*sizeof(long long));
  work.upload(d.length,length.data(),length.size()*sizeof(int));
  work.upload(d.v,v.data(),v.size()*sizeof(double));

  using Side = DeviceWork::Side;
  const Side zero{0,nullptr,0,nullptr};              // every element reads d.zero[0]
  const double* ta = d.t+d.t_offset[a]+share.first*ua*ell;              // the rows of the ABFs of the part
  if (site)                                          // D[loc(A), S] from the layout
    work.move(c.local.size(),c.longest,d.length+gathered,1.0,false,d.dloc,Side{0,d.list+local,1,nullptr},d.density,
              Side{0,d.list+stored,1,nullptr});
  else                                               // D[loc(A), .] from the rows of D
    work.move(ell,s,nullptr,1.0,false,d.dloc,Side{static_cast<long long>(s),nullptr,1,nullptr},d.density,
              Side{0,d.list+row,1,nullptr});
  work.multiply(rows,s,ell,ta,d.dloc,d.g);                            // G = T D[loc(A), .]
  work.move(rows,s,nullptr,1.0,false,d.bracket,Side{static_cast<long long>(s),nullptr,1,nullptr},d.zero,zero);

  const auto signed_size = [](std::size_t value) { return static_cast<long long>(value); };
  for (std::size_t ib = 0; ib < c.groups.size(); ++ib) {
    const DeviceGroup& group = c.groups[ib];
    const std::size_t b = group.atom, nq = system.abf_count[b], ub = system.orbital_count[b], lb = group.column.size();
    const long long* column = d.list+at[ib].column;
    work.multiply(np,ub*lb,nq,d.v+v_offset[ib],d.t+d.t_offset[b],d.w);                    // W = V T_B
    work.move(ub*np,lb,nullptr,1.0,false,d.wn,Side{signed_size(lb),nullptr,1,nullptr},d.w,
              Side{0,d.list+at[ib].w_rows,1,nullptr});                                      // Wn
    work.move(ua*np,lb,nullptr,1.0,false,d.gg,Side{signed_size(lb),nullptr,1,nullptr},d.g,
              Side{0,d.list+g_rows,1,column});                                              // Gg
    work.move(ua*np,ub,nullptr,1.0,false,d.gbp,Side{signed_size(ub),nullptr,1,nullptr},d.gg,
              Side{signed_size(lb),nullptr,1,nullptr});                                     // GBp
    work.move(ua,lb,nullptr,1.0,false,d.dl,Side{signed_size(lb),nullptr,1,nullptr},d.dloc,
              Side{signed_size(s),nullptr,1,column});                                       // D[A, loc(B)]
    work.move(ub,ua,nullptr,1.0,false,d.dba,Side{signed_size(ua),nullptr,1,nullptr},d.dloc,
              Side{0,column,signed_size(s),nullptr});                                       // D[A, B] as [nu][mu]
    work.multiply_transposed(ua,ub,np*lb,d.gg,d.wn,d.first);                                // (i)
    work.multiply(ua,lb,np*ub,d.gbp,d.w,d.second);                                          // (ii)
    work.multiply_transposed(ub*np,ua,lb,d.wn,d.dl,d.x3);                                   // (iii)
    work.transposed_multiply(ua,np*lb,ub,d.dba,d.wn,d.x4);                                  // (iv)
    work.move(ub,np*ua,nullptr,1.0,true,d.bracket,Side{0,column,signed_size(s),nullptr},d.x3,
              Side{signed_size(np*ua),nullptr,1,nullptr});                                  // bracket[., B] += X3
    work.move(ua*np,lb,nullptr,1.0,true,d.bracket,Side{0,d.list+g_rows,1,column},d.x4,
              Side{signed_size(lb),nullptr,1,nullptr});                                     // bracket[., loc(B)] += X4
    work.move(ua,ub,nullptr,factor,true,d.k+group.first,Side{group.first_stride,nullptr,1,nullptr},d.first,
              Side{signed_size(ub),nullptr,1,nullptr});                                     // K += (i)
    if (site)
      work.move(group.second_length.size(),group.second_longest,d.length+at[ib].second_length,factor,true,d.k,
                Side{0,d.list+at[ib].second_stored,1,nullptr},d.second,Side{0,d.list+at[ib].second_local,1,nullptr});
    else
      work.move(ua,lb,nullptr,factor,true,d.k+group.second,Side{group.second_stride,nullptr,1,column},d.second,
                Side{signed_size(lb),nullptr,1,nullptr});                                   // K += (ii)
  }

  work.transposed_multiply(ell,s,rows,ta,d.bracket,d.dloc);           // the last result, over D[loc(A), .]
  if (site)
    // The rows of one site of loc(A) lie in different blocks; two sites can
    // add to one block, so the sites go one after the other, as on the host.
    for (std::size_t i = 0; i+1 < c.begin.size(); ++i) {
      const std::size_t first = c.begin[i], count = c.begin[i+1]-first;
      work.move(count,c.longest,d.length+gathered+first,factor,true,d.k,Side{0,d.list+stored+first,1,nullptr},d.dloc,
                Side{0,d.list+local+first,1,nullptr});
    }
  else
    work.move(ell,s,nullptr,factor,true,d.k,Side{0,d.list+row,1,nullptr},d.dloc,
              Side{signed_size(s),nullptr,1,nullptr});
  work.finish();
}

// The parts of the centre atoms on the device; false when the device does
// not take them or stops.  Form gives the sizes and the plan of a centre and
// the T blocks of an atom.  k is written at the end only.
template <class Form>
bool device_range(const System& system, const std::vector<CentrePart>& parts, double factor,
                  std::span<const double> density, std::span<double> k, const OpenMXExxDevice& device, Form& form)
{
  if (device.accept == nullptr || device.allocate == nullptr || device.release == nullptr ||
      device.upload == nullptr || device.download == nullptr || device.dgemm == nullptr || device.move == nullptr ||
      device.finish == nullptr || parts.empty())
    return false;
  const std::size_t atoms = system.orbital_count.size();
  DeviceSizes z;
  for (const CentrePart& part : parts) z.widen(form.sizes(part.atom,part.last-part.first));
  std::vector<std::size_t> t_size(atoms);
  std::size_t t_total = 0;
  for (std::size_t x = 0; x < atoms; ++x) t_total += t_size[x] = form.t_size(x);
  if (device.accept(device.context,z.multiplications,DeviceArrays::bytes(z,k.size(),t_total)) != 1) return false;

  std::vector<double> result(k.size());
  try {
    DeviceWork work(device);
    DeviceArrays d(work,z,k.size(),t_size);
    work.upload(d.density,density.data(),density.size()*sizeof(double));
    work.upload(d.k,k.data(),k.size()*sizeof(double));
    for (std::size_t x = 0; x < atoms; ++x) work.upload(d.t+d.t_offset[x],form.t(x),t_size[x]*sizeof(double));
    const double nothing = 0.0;
    work.upload(d.zero,&nothing,sizeof(double));
    for (const CentrePart& part : parts) device_centre(system,part,factor,form.plan(part.atom),d,work);
    work.download(result.data(),d.k,result.size()*sizeof(double));
  } catch (const DeviceStop&) {
    return false;
  }
  std::copy(result.begin(),result.end(),k.begin());
  return true;
}

// The site form in one array for the device.
class DeviceBlocks {
 public:
  DeviceBlocks(const System& system, const BlockLayout& layout, double factor)
      : system_(system), factor_(factor), store_(layout,std::span<double>())
  {
  }

  // The sizes of np ABFs of centre a.
  DeviceSizes sizes(std::size_t a, std::size_t np)
  {
    const SiteSink<LayoutBlocks> sink(system_,a,factor_,store_);
    std::vector<std::array<std::size_t,3>> groups;
    for (const Site& b : system_.v_neighbours[a]) {
      const std::size_t atom = static_cast<std::size_t>(b.atom);
      groups.push_back({atom,system_.c_offset[atom].back(),system_.c_neighbours[atom].size()});
    }
    const std::size_t ell = system_.c_offset[a].back();
    return device_sizes(system_,a,np,sink.size(),ell,groups,ell*sink.site_count(),true);
  }

  std::size_t t_size(std::size_t x) const { return system_.t[x].size(); }
  const double* t(std::size_t x) const { return system_.t[x].data(); }

  DeviceCentre plan(std::size_t a)
  {
    SiteSink<LayoutBlocks> sink(system_,a,factor_,store_);
    DeviceCentre c;
    c.s = sink.size();
    sink.device_rows(c.local,c.stored,c.length,c.begin);
    for (const int length : c.length) c.longest = std::max(c.longest,static_cast<std::size_t>(length));
    for (std::size_t ib = 0; ib < sink.groups(); ++ib) {
      DeviceGroup group;
      group.atom = sink.atom(ib);
      group.v = sink.v(ib);
      for (const std::size_t column : sink.columns(ib)) group.column.push_back(static_cast<long long>(column));
      group.first = sink.device_first(ib);
      group.first_stride = static_cast<long long>(system_.orbital_count[group.atom]);
      sink.device_second(ib,group.second_stored,group.second_local,group.second_length);
      for (const int length : group.second_length)
        group.second_longest = std::max(group.second_longest,static_cast<std::size_t>(length));
      c.groups.push_back(std::move(group));
    }
    return c;
  }

 private:
  const System& system_;
  double factor_;
  LayoutBlocks store_;                     // the layout, for the offsets of the blocks; nothing is written to it
};

// The Gamma form for the device.  The T blocks are folded as GammaSink folds
// them; folding depends on the atom only.
class DeviceGamma {
 public:
  DeviceGamma(const System& system, double factor, const std::vector<std::size_t>& offset)
      : system_(system), factor_(factor), offset_(offset), folding_(system,0,factor,std::span<double>(),offset)
  {
  }

  // The sizes of np ABFs of centre a.
  DeviceSizes sizes(std::size_t a, std::size_t np)
  {
    GammaSink sink(system_,a,factor_,std::span<double>(),offset_);
    std::vector<std::array<std::size_t,3>> groups;
    for (std::size_t g = 0; g < sink.groups(); ++g) groups.push_back({sink.atom(g),sink.width(sink.atom(g)),0});
    return device_sizes(system_,a,np,offset_.back(),sink.width(a),groups,0,false);
  }

  std::size_t t_size(std::size_t x)
  {
    return system_.abf_count[x]*system_.orbital_count[x]*folding_.width(x);
  }
  const double* t(std::size_t x) { return folding_.t(x); }

  DeviceCentre plan(std::size_t a)
  {
    sink_.emplace(system_,a,factor_,std::span<double>(),offset_);  // the V blocks of the groups live in it
    GammaSink& sink = *sink_;
    const std::size_t n = offset_.back();
    DeviceCentre c;
    c.s = n;
    for (const std::size_t r : sink.rows()) c.row.push_back(static_cast<long long>(r*n));
    for (std::size_t g = 0; g < sink.groups(); ++g) {
      DeviceGroup group;
      group.atom = sink.atom(g);
      group.v = sink.v(g);
      for (const std::size_t column : sink.columns(g)) group.column.push_back(static_cast<long long>(column));
      group.first = static_cast<long long>(offset_[a]*n+offset_[group.atom]);
      group.first_stride = static_cast<long long>(n);
      group.second = static_cast<long long>(offset_[a]*n);
      group.second_stride = static_cast<long long>(n);
      c.groups.push_back(std::move(group));
    }
    return c;
  }

 private:
  const System& system_;
  double factor_;
  const std::vector<std::size_t>& offset_;
  GammaSink folding_;                      // the folded T blocks of all atoms
  std::optional<GammaSink> sink_;          // the sink of the centre of the last plan
};

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
  SiteSink<BlockMatrix> sink(system,atom,factor,k);
  centre_contribution(system,atom,0,system.abf_count[atom],sink.size(),sink.density(density).data(),sink,backend);
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

// ---- site form in one array --------------------------------------------------

std::size_t BlockLayout::find(const BlockKey& key) const noexcept
{
  const auto found = std::lower_bound(keys.begin(),keys.end(),key);
  return found != keys.end() && *found == key ? static_cast<std::size_t>(found-keys.begin()) : keys.size();
}

BlockLayout exchange_layout(const System& system)
{
  check(system);
  const std::size_t atoms = system.orbital_count.size();

  // S(A) of every centre atom, and the range of the cells of its sites.
  struct Columns {
    std::vector<Site> sites;
    std::array<int,3> low{}, high{};
  };
  std::vector<Columns> columns(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    Columns& s = columns[a];
    s.sites = column_sites(system,a);                 // begins with A itself, in the cell at the origin
    for (const Site& site : s.sites)
      for (std::size_t x = 0; x < 3; ++x) {
        s.low[x] = std::min(s.low[x],site.cell[x]);
        s.high[x] = std::max(s.high[x],site.cell[x]);
      }
  }

  // The rows of the blocks of a centre atom are its C neighbours.  For every
  // atom x: the centres that have an image of x as a row site, with the cell
  // of that image.
  struct Row {
    std::size_t centre;
    std::array<int,3> cell;
  };
  std::vector<std::vector<Row>> rows(atoms);
  for (std::size_t a = 0; a < atoms; ++a)
    for (const Site& site : system.c_neighbours[a]) rows[static_cast<std::size_t>(site.atom)].push_back({a,site.cell});

  // The keys of row atom x are (x, y, R_y - R_x) for the row sites (x, R_x)
  // and the column sites (y, R_y) of one centre.  They are marked in a table
  // over y and the cells in range and read off in the order of the keys.
  BlockLayout layout;
  layout.offset.push_back(0);
  std::vector<unsigned char> seen;
  for (std::size_t x = 0; x < atoms; ++x) {
    std::array<int,3> low{}, high{};                  // the row site (x, 0) of centre x gives the cell 0
    for (const Row& row : rows[x])
      for (std::size_t i = 0; i < 3; ++i) {
        low[i] = std::min(low[i],columns[row.centre].low[i]-row.cell[i]);
        high[i] = std::max(high[i],columns[row.centre].high[i]-row.cell[i]);
      }
    std::array<std::size_t,3> extent{};
    for (std::size_t i = 0; i < 3; ++i) extent[i] = static_cast<std::size_t>(high[i]-low[i])+1;
    seen.assign(atoms*extent[0]*extent[1]*extent[2],0);
    for (const Row& row : rows[x])
      for (const Site& site : columns[row.centre].sites) {
        std::size_t index = static_cast<std::size_t>(site.atom);
        for (std::size_t i = 0; i < 3; ++i)
          index = index*extent[i]+static_cast<std::size_t>(site.cell[i]-row.cell[i]-low[i]);
        seen[index] = 1;
      }
    std::size_t index = 0;
    for (std::size_t y = 0; y < atoms; ++y) {
      const std::size_t size = system.orbital_count[x]*system.orbital_count[y];
      for (int c0 = low[0]; c0 <= high[0]; ++c0)
        for (int c1 = low[1]; c1 <= high[1]; ++c1)
          for (int c2 = low[2]; c2 <= high[2]; ++c2, ++index) {
            if (!seen[index]) continue;
            layout.keys.push_back(BlockKey{static_cast<int>(x),static_cast<int>(y),{c0,c1,c2}});
            layout.offset.push_back(layout.offset.back()+size);
          }
    }
  }
  return layout;
}

namespace {

void check_arrays(const System& system, const BlockLayout& layout, std::span<const double> density,
                  std::span<const double> k)
{
  check(system);
  check(system,layout);
  if (density.size() != layout.elements()) throw std::invalid_argument("the density matrix does not have the size of the layout");
  if (k.size() != layout.elements()) throw std::invalid_argument("the exchange matrix does not have the size of the layout");
}

// The part of a centre atom on the host, for arrays that fit the system.
void add_blocks_part(const System& system, const BlockLayout& layout, const CentrePart& part, double factor,
                     std::span<const double> density, std::span<double> k, const DenseBackend& backend)
{
  LayoutBlocks store(layout,k);
  SiteSink<LayoutBlocks> sink(system,part.atom,factor,store);
  const std::unique_ptr<double[]> dloc = sink.density(density,backend,piece_count(part.last-part.first,backend));
  centre_contribution(system,part.atom,part.first,part.last,sink.size(),dloc.get(),sink,backend);
}

} // namespace

void add_exchange_blocks(const System& system, const BlockLayout& layout, std::size_t atom, double factor,
                         std::span<const double> density, std::span<double> k, const DenseBackend& backend)
{
  check(system);
  if (atom >= system.orbital_count.size()) throw std::out_of_range("the centre atom does not exist");
  check_arrays(system,layout,density,k);
  add_blocks_part(system,layout,CentrePart{atom,0,system.abf_count[atom]},factor,density,k,backend);
}

bool add_exchange_blocks_abfs(const System& system, const BlockLayout& layout, std::size_t first, std::size_t last,
                              double factor, std::span<const double> density, std::span<double> k,
                              const DenseBackend& backend)
{
  check(system);
  const std::vector<std::size_t> offset = abf_offsets(system);
  if (first > last || last > offset.back()) throw std::out_of_range("the range of ABFs is not inside the unit cell");
  if (first == last) return false;
  check_arrays(system,layout,density,k);
  const std::vector<CentrePart> parts = centre_parts(offset,first,last);
  if (const OpenMXExxDevice* device = backend.device(); device != nullptr) {
    DeviceBlocks form(system,layout,factor);
    if (device_range(system,parts,factor,density,k,*device,form)) return true;
  }
  for (const CentrePart& part : parts) add_blocks_part(system,layout,part,factor,density,k,backend);
  return false;
}

bool add_exchange_blocks_range(const System& system, const BlockLayout& layout, std::size_t first, std::size_t last,
                               double factor, std::span<const double> density, std::span<double> k,
                               const DenseBackend& backend)
{
  if (first > last || last > system.orbital_count.size())
    throw std::out_of_range("the range of centre atoms is not inside the unit cell");
  if (first == last) return false;
  check(system);
  const std::vector<std::size_t> offset = abf_offsets(system);
  return add_exchange_blocks_abfs(system,layout,offset[first],offset[last],factor,density,k,backend);
}

// ---- the work of a build ------------------------------------------------------

std::vector<double> abf_costs(const System& system, bool sites)
{
  check(system);
  const std::size_t atoms = system.orbital_count.size(), n = orbital_total(system);
  const auto product = [](std::size_t a, std::size_t b, std::size_t c) {
    return static_cast<double>(a)*static_cast<double>(b)*static_cast<double>(c);
  };
  // The columns of the T blocks of every atom: the orbitals of its C
  // neighbours, and in the Gamma form those of the atoms among them.
  std::vector<std::size_t> width(atoms);
  std::vector<unsigned char> seen;
  for (std::size_t a = 0; a < atoms; ++a) {
    width[a] = system.c_offset[a].back();
    if (sites) continue;
    seen.assign(atoms,0);
    width[a] = 0;
    for (const Site& site : system.c_neighbours[a]) {
      const std::size_t c = static_cast<std::size_t>(site.atom);
      if (!seen[c]) width[a] += system.orbital_count[c];
      seen[c] = 1;
    }
  }
  std::vector<double> cost(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t ua = system.orbital_count[a];
    std::size_t s = n;                                 // the columns of D[loc(A), .]
    if (sites) {
      s = 0;
      for (const Site& site : column_sites(system,a)) s += system.orbital_count[static_cast<std::size_t>(site.atom)];
    }
    double total = 2.0*product(ua,s,width[a]);         // G and the last product
    // W and the four terms of every group: a V neighbour in the site form,
    // an atom among the V neighbours in the Gamma form.
    seen.assign(atoms,0);
    for (const Site& site : system.v_neighbours[a]) {
      const std::size_t b = static_cast<std::size_t>(site.atom);
      if (!sites && seen[b]) continue;
      seen[b] = 1;
      total += product(system.orbital_count[b],width[b],system.abf_count[b]+4*ua);
    }
    cost[a] = total;
  }
  return cost;
}

double block_asymmetry(const System& system, const BlockLayout& layout, std::span<const double> values)
{
  if (values.size() != layout.elements()) throw std::invalid_argument("the array does not have the size of the layout");
  double largest = 0.0;
  for_mirrored(system,layout,[&](std::size_t e, std::size_t f) {
    largest = std::max(largest,std::fabs(values[e]-values[f]));
  });
  return largest;
}

void symmetrize_blocks(const System& system, const BlockLayout& layout, std::span<double> values)
{
  if (values.size() != layout.elements()) throw std::invalid_argument("the array does not have the size of the layout");
  for_mirrored(system,layout,[&](std::size_t e, std::size_t f) {
    const double mean = 0.5*(values[e]+values[f]);
    values[e] = mean;
    values[f] = mean;
  });
}

// ---- Gamma form ------------------------------------------------------------

std::size_t orbital_total(const System& system) noexcept
{
  std::size_t total = 0;
  for (const std::size_t u : system.orbital_count) total += u;
  return total;
}

namespace {

// The part of a centre atom on the host, for N x N arrays of the system.
void add_gamma_part(const System& system, const std::vector<std::size_t>& offset, const CentrePart& part, double factor,
                    std::span<const double> density, std::span<double> k, const DenseBackend& backend)
{
  const std::size_t n = offset.back();
  GammaSink sink(system,part.atom,factor,k,offset);
  const std::unique_ptr<double[]> dloc =
      gathered_rows(density,n,sink.rows(),backend,piece_count(part.last-part.first,backend));
  centre_contribution(system,part.atom,part.first,part.last,n,dloc.get(),sink,backend);
}

} // namespace

void add_exchange_gamma(const System& system, std::size_t atom, double factor, std::span<const double> density,
                        std::span<double> k, const DenseBackend& backend)
{
  check(system);
  if (atom >= system.orbital_count.size()) throw std::out_of_range("the centre atom does not exist");
  const std::vector<std::size_t> offset = orbital_offsets(system);
  const std::size_t n = offset.back();
  check_dense(n,density.size(),"the density matrix is not N x N");
  check_dense(n,k.size(),"the exchange matrix is not N x N");
  add_gamma_part(system,offset,CentrePart{atom,0,system.abf_count[atom]},factor,density,k,backend);
}

bool add_exchange_gamma_abfs(const System& system, std::size_t first, std::size_t last, double factor,
                             std::span<const double> density, std::span<double> k, const DenseBackend& backend)
{
  check(system);
  const std::vector<std::size_t> abfs = abf_offsets(system);
  if (first > last || last > abfs.back()) throw std::out_of_range("the range of ABFs is not inside the unit cell");
  if (first == last) return false;
  const std::vector<std::size_t> offset = orbital_offsets(system);
  const std::size_t n = offset.back();
  check_dense(n,density.size(),"the density matrix is not N x N");
  check_dense(n,k.size(),"the exchange matrix is not N x N");
  const std::vector<CentrePart> parts = centre_parts(abfs,first,last);
  if (const OpenMXExxDevice* device = backend.device(); device != nullptr) {
    DeviceGamma form(system,factor,offset);
    if (device_range(system,parts,factor,density,k,*device,form)) return true;
  }
  for (const CentrePart& part : parts) add_gamma_part(system,offset,part,factor,density,k,backend);
  return false;
}

bool add_exchange_gamma_range(const System& system, std::size_t first, std::size_t last, double factor,
                              std::span<const double> density, std::span<double> k, const DenseBackend& backend)
{
  if (first > last || last > system.orbital_count.size())
    throw std::out_of_range("the range of centre atoms is not inside the unit cell");
  if (first == last) return false;
  check(system);
  const std::vector<std::size_t> abfs = abf_offsets(system);
  return add_exchange_gamma_abfs(system,abfs[first],abfs[last],factor,density,k,backend);
}

std::vector<double> exchange_gamma(const System& system, std::span<const double> density, double alpha,
                                   const DenseBackend& backend)
{
  const std::size_t n = orbital_total(system);
  std::vector<double> k(n*n,0.0);
  add_exchange_gamma_range(system,0,system.orbital_count.size(),-alpha,density,k,backend);
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

CentreGradient centre_gradient_blocks(const System& system, const BlockLayout& layout, std::size_t atom, double factor,
                                      std::span<const double> density, const DenseBackend& backend)
{
  check(system);
  if (atom >= system.orbital_count.size()) throw std::out_of_range("the centre atom does not exist");
  return centre_gradient_blocks(system,layout,atom,0,system.abf_count[atom],factor,density,backend);
}

CentreGradient centre_gradient_blocks(const System& system, const BlockLayout& layout, std::size_t atom,
                                      std::size_t first, std::size_t last, double factor,
                                      std::span<const double> density, const DenseBackend& backend)
{
  check(system);
  if (atom >= system.orbital_count.size()) throw std::out_of_range("the centre atom does not exist");
  if (first > last || last > system.abf_count[atom])
    throw std::out_of_range("the range of ABFs is not inside the centre atom");
  check(system,layout);
  if (density.size() != layout.elements()) throw std::invalid_argument("the density matrix does not have the size of the layout");
  CentreGradient gradient;
  if (first == last) {
    gradient.v.resize(system.v_neighbours[atom].size());
    return gradient;
  }
  const DensityBlocks blocks = [&layout,density](const BlockKey& key, std::span<double> block) {
    const std::size_t i = layout.find(key);
    if (i == layout.keys.size()) throw std::invalid_argument("the layout lacks a block of the density matrix");
    std::copy_n(density.begin()+static_cast<std::ptrdiff_t>(layout.offset[i]),block.size(),block.begin());
  };
  // The sink gives the blocks and the columns; nothing is written to K.
  LayoutBlocks store(layout,std::span<double>());
  SiteSink<LayoutBlocks> sink(system,atom,factor,store);
  centre_derivatives(system,atom,first,last,sink.size(),sink.density(blocks),sink,factor,backend,gradient.t,gradient.v);
  return gradient;
}

CentreGradient centre_gradient_gamma(const System& system, std::size_t atom, double factor,
                                     std::span<const double> density, const DenseBackend& backend)
{
  check(system);
  if (atom >= system.orbital_count.size()) throw std::out_of_range("the centre atom does not exist");
  return centre_gradient_gamma(system,atom,0,system.abf_count[atom],factor,density,backend);
}

CentreGradient centre_gradient_gamma(const System& system, std::size_t atom, std::size_t first, std::size_t last,
                                     double factor, std::span<const double> density, const DenseBackend& backend)
{
  check(system);
  if (atom >= system.orbital_count.size()) throw std::out_of_range("the centre atom does not exist");
  if (first > last || last > system.abf_count[atom])
    throw std::out_of_range("the range of ABFs is not inside the centre atom");
  const std::vector<std::size_t> offset = orbital_offsets(system);
  const std::size_t n = offset.back();
  check_dense(n,density.size(),"the density matrix is not N x N");
  CentreGradient gradient;
  const std::vector<Site>& neighbours = system.v_neighbours[atom];
  if (first == last) {
    gradient.v.resize(neighbours.size());
    return gradient;
  }
  GammaSink sink(system,atom,factor,std::span<double>(),offset);
  std::vector<double> folded_t;
  std::vector<std::vector<double>> grouped_v;
  const std::vector<std::size_t> row = sink.rows();
  centre_derivatives(system,atom,first,last,n,gathered_rows(density,n,row),sink,factor,backend,folded_t,grouped_v);

  // The stored blocks: the images of an atom among the C neighbours share a
  // column of the folded T blocks, the images among the V neighbours a group.
  const std::size_t rows = (last-first)*system.orbital_count[atom];
  const std::size_t ell = system.c_offset[atom].back(), width = row.size();
  std::vector<std::size_t> position(n,0);
  for (std::size_t l = 0; l < width; ++l) position[row[l]] = l;
  const std::vector<std::size_t> stored = folded_columns(system,offset,atom);
  gradient.t.resize(rows*ell);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t l = 0; l < ell; ++l) gradient.t[r*ell+l] = folded_t[r*width+position[stored[l]]];
  std::vector<std::size_t> group(system.orbital_count.size(),0);
  for (std::size_t g = 0; g < sink.groups(); ++g) group[sink.atom(g)] = g;
  gradient.v.resize(neighbours.size());
  for (std::size_t ib = 0; ib < neighbours.size(); ++ib)
    gradient.v[ib] = grouped_v[group[static_cast<std::size_t>(neighbours[ib].atom)]];
  return gradient;
}

} // namespace openmx::exx
