#pragma once

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

inline std::string pao_fixture(bool energies = true, double atom_species = 6.25)
{
  const std::array<double,8> radii{.1,.2,.4,.7,1.1,1.8,2.5,4.};
  std::ostringstream text;
  text << std::setprecision(17)
       << "# Synthetic PAO: analytic quadratic radial channels, s through g.\n"
       << "ignored.generator.option 123\nAtomSpecies " << atom_species
       << "\ngrid.num.output 8\nradial.cutoff.pao 6\n";
  if (energies) {
    text << "Lmax= 4 Mul=2\n";
    for (int l = 0; l <= 4; ++l)
      for (int mu = 0; mu < 2; ++mu)
        text << "l mu " << l << ' ' << mu << ' ' << -.5*(l+1)/(mu+1) << '\n';
  }
  text << "<valence.charge.density\n";
  for (std::size_t i = 0; i < radii.size(); ++i)
    text << std::log(radii[i]) << ' ' << radii[i] << ' ' << .125*(i+1) << " # density\n";
  text << "valence.charge.density>\nPAO.Lmax 4\nPAO.Mul 2\n";
  for (int l = 0; l <= 4; ++l) {
    text << "<pseudo.atomic.orbitals.L=" << l << '\n';
    for (double r : radii) {
      text << std::log(r) << ' ' << r;
      for (int mu = 0; mu < 2; ++mu) text << ' ' << (l+1)*(mu+1)*(2+3*r+4*r*r);
      text << '\n';
    }
    text << "pseudo.atomic.orbitals.L=" << l << ">\n";
  }
  return text.str();
}
