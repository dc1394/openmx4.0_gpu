// The fixture tests/fixtures/kbuild_reference.txt, written by
// "tools/exx_reference.py --dump": a model of two atoms in a cell that is
// smaller than its cutoffs, with random T and V blocks, the blocks of a random
// density matrix, and the blocks of K and the energy that the reference
// implementation of section 6.4 gives for them.  Section 10.3 of the
// implementation guide describes the format.
#pragma once

#include <openmx/exx/exchange.hpp>

#include <map>
#include <string>
#include <vector>

namespace exx_test {

struct KbuildFixture {
  double alpha = 0.0;
  // Sizes, neighbour lists, T and V blocks; the report is not used.
  openmx::exx::System system;
  // D_xy(R): every block that the reference implementation read.
  std::map<openmx::exx::BlockKey,std::vector<double>> density;
  // K_xy(R) of one spin channel.
  std::map<openmx::exx::BlockKey,std::vector<double>> k;
  double energy = 0.0; // equation (4), one spin channel
};

KbuildFixture read_kbuild_fixture(const std::string& path);

} // namespace exx_test
