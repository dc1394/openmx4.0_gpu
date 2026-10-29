#include <openmx/basis/pao_file.hpp>
#include <iostream>

int main(int argc, char** argv)
{
  try {
    if (argc != 2) { std::cerr << "usage: pao_example FILE.pao\n"; return 1; }
    const auto pao = openmx::basis::PaoFile::from_file(argv[1]);
    const auto& m = pao.metadata();
    std::cout << "AtomSpecies=" << m.atom_species << " lmax=" << m.lmax
              << " multiplicity=" << m.multiplicity << " points=" << m.mesh_points << '\n';
    std::cout << "R(l=0, mu=0, r=0.85 bohr)=" << pao.sample(0,0,.85) << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
