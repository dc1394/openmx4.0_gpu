// Reads an OpenMX-style keyword file: the file named on the command line, or
// a small one written into the working directory when none is given.
#include <openmx/io/keyword_file.hpp>

#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char** argv)
{
  std::string path = argc > 1 ? argv[1] : "keywords_example.dat";
  if (argc < 2) {
    std::ofstream out(path);
    out << "System.Name        water\n"
           "scf.maxIter        60\n"
           "scf.energycutoff   180.0\n"
           "scf.Kgrid          4 4 1\n"
           "<Atoms.SpeciesAndCoordinates\n"
           " 1 O 0.000 0.000 0.000 3.0 3.0\n"
           " 2 H 0.757 0.586 0.000 0.5 0.5\n"
           " 3 H -0.757 0.586 0.000 0.5 0.5\n"
           "Atoms.SpeciesAndCoordinates>\n";
  }
  openmx::io::KeywordFile file(path);   // throws when the file cannot be opened
  std::string name;
  int iterations = 0, kgrid[3] = {1, 1, 1};
  double cutoff = 0.0;
  file.string("System.Name", name, "unnamed");
  file.integer("scf.maxIter", iterations, 40);
  file.real("scf.energycutoff", cutoff, 150.0);
  const int ones[3] = {1, 1, 1};
  file.integers("scf.Kgrid", kgrid, ones);
  std::printf("%s: %d iterations, cutoff %.1f Ry, k-grid %d %d %d\n",
              name.c_str(), iterations, cutoff, kgrid[0], kgrid[1], kgrid[2]);
  if (const auto atoms = file.block("Atoms.SpeciesAndCoordinates"))
    std::printf("%zu atom lines; errors recorded: %d\n", atoms->size(), file.error_count());
  return 0;
}
