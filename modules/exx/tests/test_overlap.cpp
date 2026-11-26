// exx-two-center-overlap: equation (12) with the overlap kernel, applied to
// PAOs, against the overlap matrix OLP[0] of the main program.  The fixture
// (tests/fixtures/overlap_reference.txt, written by
// tools/exx_overlap_fixture.py) holds the radial functions of the PAOs, the
// geometry and blocks of the overlap matrix of a run of the main program.
//
// The comparison checks the conventions that the exact exchange shares with
// the main program: the order of the orbitals, the signs and the order of the
// real spherical harmonics, and the direction of the displacement.
#include <openmx/exx/two_center.hpp>

#include <openmx/basis/radial_table.hpp>

#include "overlap_fixture.hpp"
#include "test_support.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace exx_test {

using namespace openmx::exx;

void two_center_overlap(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);

  // The discretization of the main program.  FT_PAO.c samples every PAO on a
  // uniform radial grid from the first mesh point to the cutoff radius plus
  // 0.5 bohr and stores its transform at k_j = j kmax/NumGridK,
  // j = 0..NumGridK-1.  Set_OLP_Kin.c integrates over
  // kappa_i = kmin + i (kmax-kmin)/NumGridR with the trapezoid rule and reads
  // the transforms from that table by interpolation.
  const ReciprocalGrid quadrature = ReciprocalGrid::uniform(fixture.kmax,fixture.radial_intervals,fixture.kmin);
  const double table_step = fixture.kmax/fixture.reciprocal_points;
  const ReciprocalGrid table_nodes =
      ReciprocalGrid::uniform(table_step*(fixture.reciprocal_points-1),fixture.reciprocal_points-1);

  int maximum = 0;
  std::vector<std::vector<ReciprocalFunction>> direct(fixture.species.size()), tabulated(fixture.species.size());
  for (std::size_t s = 0; s < fixture.species.size(); ++s) {
    const FixtureSpecies& species = fixture.species[s];
    const RadialGrid grid{species.radii.front(),
                          (species.cutoff+0.5-species.radii.front())/fixture.radial_intervals,fixture.radial_intervals};
    const std::vector<RadialFunction> radial = sample_shells(species,grid);
    for (const RadialFunction& f : radial) maximum = std::max(maximum,f.angular);
    // Transforms at the quadrature nodes, and the route of the main program.
    direct[s] = bessel_transform(grid,radial,quadrature);
    const std::vector<ReciprocalFunction> table = bessel_transform(grid,radial,table_nodes);
    for (const ReciprocalFunction& row : table) {
      ReciprocalFunction f{row.angular,std::vector<double>(quadrature.size())};
      for (std::size_t i = 0; i < quadrature.size(); ++i)
        f.values[i] = openmx::basis::sample_reciprocal_row(openmx::basis::ReciprocalKind::pao,row.values,table_step,
                                                           quadrature.nodes()[i]);
      tabulated[s].push_back(std::move(f));
    }
  }

  const TwoCenterIntegrator integrator(quadrature,overlap_kernel(),maximum,maximum);
  Worst legacy("overlap matrix of the main program, its discretization",1e-8);
  Worst plain("the same with transforms at the quadrature nodes",1e-6);
  double largest_offsite = 0.0;
  long offsite = 0;
  for (const FixturePair& pair : fixture.pairs) {
    const FixtureSpecies& sa = fixture.species[static_cast<std::size_t>(fixture.atom_species[static_cast<std::size_t>(pair.a)])];
    const FixtureSpecies& sb = fixture.species[static_cast<std::size_t>(fixture.atom_species[static_cast<std::size_t>(pair.b)])];
    const std::size_t ia = static_cast<std::size_t>(fixture.atom_species[static_cast<std::size_t>(pair.a)]);
    const std::size_t ib = static_cast<std::size_t>(fixture.atom_species[static_cast<std::size_t>(pair.b)]);
    // d = tau_b + R - tau_a: from the atom of the rows to the atom of the columns.
    std::array<double,3> d{};
    for (int x = 0; x < 3; ++x) {
      d[static_cast<std::size_t>(x)] = fixture.positions[static_cast<std::size_t>(pair.b)][static_cast<std::size_t>(x)]-
                                       fixture.positions[static_cast<std::size_t>(pair.a)][static_cast<std::size_t>(x)];
      for (int v = 0; v < 3; ++v)
        d[static_cast<std::size_t>(x)] += pair.cell[static_cast<std::size_t>(v)]*
                                          fixture.lattice[static_cast<std::size_t>(v)][static_cast<std::size_t>(x)];
    }
    const auto geometry = integrator.pair(d);
    const bool onsite = pair.a == pair.b && pair.cell == std::array<int,3>{0,0,0};
    int row = 0;
    for (std::size_t p = 0; p < sa.shells.size(); ++p) {
      const int rows = 2*sa.shells[p].l+1;
      int column = 0;
      for (std::size_t q = 0; q < sb.shells.size(); ++q) {
        const int columns = 2*sb.shells[q].l+1;
        std::vector<double> block(static_cast<std::size_t>(rows*columns)), block_plain(block.size());
        geometry.block(tabulated[ia][p],tabulated[ib][q],block);
        geometry.block(direct[ia][p],direct[ib][q],block_plain);
        for (int i = 0; i < rows; ++i)
          for (int j = 0; j < columns; ++j) {
            const double reference = pair.values[static_cast<std::size_t>((row+i)*pair.columns+column+j)];
            legacy.absolute(block[static_cast<std::size_t>(i*columns+j)],reference);
            plain.absolute(block_plain[static_cast<std::size_t>(i*columns+j)],reference);
            if (!onsite) { largest_offsite = std::fmax(largest_offsite,std::fabs(reference)); ++offsite; }
          }
        column += columns;
      }
      require(column == pair.columns,"the columns of an overlap block do not match the shells");
      row += rows;
    }
    require(row == pair.rows,"the rows of an overlap block do not match the shells");
  }
  legacy.finish();
  plain.finish();
  std::printf("off-site elements: %ld, largest magnitude %.4f\n",offsite,largest_offsite);
  require(offsite > 1000 && largest_offsite > 0.3,"the overlap fixture has no sizeable off-site elements");
}

} // namespace exx_test
