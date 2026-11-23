#pragma once

#include <openmx/exx/kernel.hpp>
#include <openmx/exx/lri.hpp>
#include <openmx/exx/two_center.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <vector>

// The quantities that depend on the structure, section 6.2 of the
// implementation guide: sites, the two neighbour lists, and the T and V
// blocks.  Nothing here uses MPI; one call prepares the whole unit cell.
namespace openmx::exx {

// An atom of the unit cell together with the lattice vector of its cell,
// R = cell[0] a1 + cell[1] a2 + cell[2] a3.  Its position is tau_atom + R.
struct Site {
  int atom = 0;
  std::array<int,3> cell{};
  friend bool operator==(const Site&, const Site&) = default;
};

// The unit cell and its atoms.  Lengths are in bohr, positions Cartesian.
struct Structure {
  std::array<std::array<double,3>,3> lattice{}; // lattice[i] is the vector a_(i+1)
  std::vector<std::array<double,3>> positions;
  std::vector<int> species;                     // index of the species of every atom
};

// tau_site.atom + R - tau_atom: from the atom in the cell at the origin to the
// site.  The site (b, R) seen from a and the site (a, -R) seen from b give
// vectors that differ in sign only, to the last bit.
std::array<double,3> displacement(const Structure& structure, int atom, const Site& site);

// For every atom a of the unit cell: the sites (b, R) whose distance from a
// does not exceed cutoffs[species_a*species_count + species_b].  The atom
// itself comes first; the other sites follow ordered by atom, then by the
// three components of the cell.  The order does not depend on rounding.  The
// lattice vectors are enumerated here and are not limited by any table of
// the main program.  cutoffs must be symmetric.
std::vector<std::vector<Site>> neighbour_lists(const Structure& structure, std::span<const double> cutoffs,
                                               std::size_t species_count);

// Position of a site in a list made by neighbour_lists(); the size of the list
// if it is absent.
std::size_t find_site(std::span<const Site> list, const Site& site);

struct SystemOptions {
  // The interaction v of the exchange: short_range_kernel(omega) for HSE06
  // (the default, with omega = 0.11/bohr), truncated_coulomb_kernel(R_c) for
  // PBE0.
  Kernel interaction = short_range_kernel(0.11);
  // scf.exx.v.cutoff: V blocks are kept up to the larger ABF radius of the two
  // atoms plus this length (bohr).
  double v_cutoff = 8.0;
  LriOptions lri;
};

// Neighbour lists and blocks of one structure.
struct System {
  std::vector<std::size_t> orbital_count; // u_a for every atom
  std::vector<std::size_t> abf_count;     // n_P,a for every atom
  // C neighbours: the sites whose PAOs overlap those of the atom (distance up
  // to the sum of the two PAO radii).  V neighbours: the sites whose V block
  // is kept.  Both lists start with the atom itself.
  std::vector<std::vector<Site>> c_neighbours, v_neighbours;
  // The T blocks of an atom are stored side by side, as section 6.4 uses
  // them: c_offset[a][i] is the first column of the i-th C neighbour of atom a,
  // and c_offset[a].back() is the number of columns ell_a, the number of
  // orbitals on all C neighbours.
  std::vector<std::vector<std::size_t>> c_offset;
  // t[a][(P*u_a + mu)*ell_a + c_offset[a][i] + lambda] = C^(1)P_{mu lambda}:
  // P and mu on atom a, lambda on its i-th C neighbour.  The columns of the
  // atom itself (i = 0) hold half of the solution of equation (7) for the
  // atom alone.
  std::vector<std::vector<double>> t;
  // v[a][i][P*n_b + Q] = (P|v|Q): P on atom a, Q on its i-th V neighbour (atom b).
  std::vector<std::vector<std::vector<double>>> v;
  LriReport report;
};

// The T block of an atom and one of its C neighbours as a separate array,
// block[(P*u_a + mu)*u_c + lambda].
std::vector<double> t_block(const System& system, std::size_t atom, std::size_t neighbour);

// Section 6.2 for one structure.  species[s] is the table of species s; all
// tables must have been made on "reciprocal".  The ABF radius of a species is
// the radius of its table.  Each pair of sites is solved once and its solution
// goes into the T blocks of both atoms; a V block is computed once and its
// transpose goes to the other atom.
System prepare_system(const Structure& structure, std::span<const SpeciesTable> species,
                      const ReciprocalGrid& reciprocal, const SystemOptions& options);

} // namespace openmx::exx
