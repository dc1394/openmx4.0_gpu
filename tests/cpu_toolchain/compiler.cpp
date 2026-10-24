#include <numeric>
#include <span>
#include <vector>

extern "C" int openmx_test_cpu_cxx()
{
    const std::vector<int> values{1, 2, 3};
    const std::span<const int> view(values);
    return std::accumulate(view.begin(), view.end(), 0);
}
