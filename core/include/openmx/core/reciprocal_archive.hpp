#pragma once
#include "openmx_core_reciprocal_archive.h"
#include <span>
#include <vector>

namespace openmx::core {

// Immutable row plan. Storage and its calculation context must outlive it;
// topology, species counts and input pointer tables do not need to survive.
// No process-global state, filesystem state or numeric row ownership here.
class ReciprocalArchive {
 public:
  ReciprocalArchive(const OpenMXReciprocalViews& tables,const OpenMXReciprocalArchiveInput& input);
  OpenMXReciprocalRows rows() const noexcept;
  void read(const char *path);
  void write(const char *path) const;
  ReciprocalArchive(const ReciprocalArchive&)=delete;
  ReciprocalArchive& operator=(const ReciprocalArchive&)=delete;
  ReciprocalArchive(ReciprocalArchive&&)=delete;
  ReciprocalArchive& operator=(ReciprocalArchive&&)=delete;
 private:
  void append(double *values,int count);
  std::vector<std::size_t> offsets_{0};
  std::vector<OpenMXReciprocalRow> rows_;
  std::size_t bytes_=0;
};

} // namespace openmx::core
