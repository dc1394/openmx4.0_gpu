// Portable end-to-end transform benchmark: sampling is prepared once, while
// each timed call includes device allocations, transfers and host results.
#include <openmx/core/radial.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace openmx::core;
using Clock = std::chrono::steady_clock;
struct Shape {
  const char* name;
  int lmax, points, waves, orbitals;
  bool products;
};

int integer(std::string_view text)
{
  int value;
  const auto [end,error] = std::from_chars(text.data(),text.data()+text.size(),value);
  if (error != std::errc{} || end != text.data()+text.size())
    throw std::invalid_argument("expected an integer");
  return value;
}

RadialSamples make_samples(Shape shape)
{
  RadialQuadrature grid{shape.lmax,{},{},{},6.0/(shape.points-1)};
  for (int i = 0; i < shape.points; ++i) {
    const double r = i*grid.step;
    grid.radii.push_back(r);
    grid.measures.push_back(r*r);
  }
  for (int k = 0; k < shape.waves; ++k) grid.wavevectors.push_back(k*0.04);
  RadialSamples samples{std::move(grid),{},Tensor<double,2>({
    static_cast<std::size_t>(shape.orbitals),static_cast<std::size_t>(shape.points)}),{}};
  for (int orbital = 0; orbital < shape.orbitals; ++orbital) {
    samples.orbitals.push_back({orbital%(shape.lmax+1),orbital/(shape.lmax+1)});
    for (int i = 0; i < shape.points; ++i)
      element(samples.values.view(),orbital,i) =
        std::exp(-samples.grid.radii[i]*(1+0.1*orbital));
  }
  if (shape.products)
    for (int a = 0; a < shape.orbitals; ++a)
      for (int b = a; b < shape.orbitals; ++b)
        for (int l = 0; l <= shape.lmax; ++l) samples.products.push_back({a,b,l});
  return samples;
}
} // namespace

int main(int argc, char** argv)
{
  try {
    int repeats = 9, device = has_accelerator() ? 0 : -1;
    for (int i = 1; i < argc; ++i) {
      const std::string_view option = argv[i];
      if (option == "--help") {
        std::puts("openmx_radial_benchmark [--repetitions N] [--device N]\n"
                  "device -1 selects the host path; two warmups precede each case.");
        return 0;
      }
      if ((option != "--repetitions" && option != "--device") || i+1 == argc)
        throw std::invalid_argument("use --repetitions N and/or --device N");
      const int value = integer(argv[++i]);
      if (option == "--device") device = value;
      else repeats = value;
    }
    if (repeats < 1 || device < -1)
      throw std::invalid_argument("repetitions must be positive and device must be >= -1");
    if (device >= 0 && !has_accelerator())
      throw std::invalid_argument("this build has no accelerator backend; use --device -1");
    Context context;
    for (const auto shape : {
        Shape{"neutral",0,901,900,1,false},
        Shape{"pao",3,901,900,8,false},
        Shape{"products",6,200,500,8,true},
        Shape{"tail",3,65,257,8,false}}) {
      const auto samples = make_samples(shape);
      const auto reference = transform_host(samples);
      double error = 0, checksum = 0;
      std::vector<double> times;
      for (int pass = -2; pass < repeats; ++pass) {
        const auto begin = Clock::now();
        const auto output = context.transform(samples,device);
        const double elapsed = std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
        checksum = 0;
        for (std::size_t i = 0; i < output.size(); ++i) {
          if (!std::isfinite(output.data()[i]))
            throw std::runtime_error("nonfinite transform result");
          error = std::max(error,std::abs(reference.data()[i]-output.data()[i]));
          checksum += output.data()[i];
        }
        if (pass >= 0) times.push_back(elapsed);
      }
      if (error > 2e-12) throw std::runtime_error("transform differs from host reference by > 2e-12");
      std::sort(times.begin(),times.end());
      std::printf("{\"case\":\"%s\",\"device\":%d,\"median_ms\":%.9f,"
        "\"minimum_ms\":%.9f,\"maximum_ms\":%.9f,\"max_error\":%.17g,"
        "\"checksum\":%.17g,\"repeats\":%d}\n",shape.name,device,
        times[times.size()/2],times.front(),times.back(),error,checksum,repeats);
      std::fflush(stdout);
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr,"radial benchmark: %s\n",error.what());
    return 1;
  }
}
