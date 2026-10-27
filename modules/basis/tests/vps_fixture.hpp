#pragma once

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

inline std::string vps_fixture(bool j_dependent = true, bool partial_core = true,
                               bool legacy = false, bool empty = false, double species = 6.25)
{
  if (legacy) j_dependent = false;
  const int channels = empty ? 0 : 3;
  const std::array<int,3> angular{0,1,4};
  const std::array<double,8> radii{.1,.2,.4,.8,1.2,1.6,2.,3.};
  std::ostringstream out;out << std::setprecision(17);
  out << "# Synthetic analytic VPS data; no data-set dependency\nAtomSpecies " << species
      << "\ntotal.electron 6\nvalence.electron 4\ngrid.num.output 8\n"
      << "j.dependent.pseudo.potentials " << (j_dependent ? "on" : "off")
      << "\ncharge.pcc.calc " << (partial_core ? "on" : "off") << '\n';
  if (legacy) {
    out << "number.vps " << (channels+1) << "\nlocal.part.vps 0\n<pseudo.NandL\n0 1 2 5\n";
    for (int i = 0; i < channels; ++i) out << i+1 << " 2 " << angular[i] << " 5\n";
    out << "pseudo.NandL>\n";
    if (channels) {
      out << "<projection.energies\n";
      for (int i = 0; i < channels; ++i) out << i+7 << ' ' << -.5*(i+1) << '\n';
      out << "projection.energies>\n";
    }
  } else {
    out << "<project.energies\n" << channels << '\n';
    for (int i = 0; i < channels; ++i) {
      out << angular[i] << ' ' << -.5*(i+1);
      if (j_dependent) out << ' ' << -.25*(i+1);
      out << '\n';
    }
    out << "project.energies>\n";
  }
  out << "<Pseudo.Potentials\n";
  for (double r : radii) {
    out << std::log(r) << ' ' << r << ' ' << -4/(1+r);
    for (int i = 0; i < channels; ++i)
      for (int spin = 0; spin < 1+j_dependent; ++spin)
        out << ' ' << (i+1)*(spin+1)*(2+3*r+4*r*r);
    out << '\n';
  }
  out << "Pseudo.Potentials>\n";
  if (partial_core) {
    out << "<density.PCC\n";
    for (std::size_t i = 0; i < radii.size(); ++i)
      out << std::log(radii[i]) << ' ' << radii[i] << ' ' << .125*(i+1) << '\n';
    out << "density.PCC>\n";
  }
  return out.str();
}
