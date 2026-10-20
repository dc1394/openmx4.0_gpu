#pragma once

#include <openmx/core/hvna_workspace.hpp>

namespace openmx::core {

struct ProjectorAtomBlock {
  int orbitals, neighbors; // Includes the self neighbor.
};

// Offset tables for a rank's local or received direction-0 projector rows.
// Atom bases are one-based at the legacy boundary; numeric rows are flat.
class ProjectorArchiveLayout {
 public:
  ProjectorArchiveLayout(int components, std::span<const ProjectorAtomBlock> atoms);
  std::span<int> bases() noexcept { return bases_; }
  std::span<std::size_t> offsets() noexcept { return offsets_; }
  std::size_t elements() const noexcept { return elements_; }
  std::size_t rows() const noexcept { return rows_; }
 private:
  std::vector<int> bases_;
  std::vector<std::size_t> offsets_;
  std::size_t elements_ = 0, rows_ = 0;
};

// An invocation-owned snapshot, independent of preparation and MPI/GPU state.
// The large numeric archives remain under each backend's memory policy.
class HvnaLayout {
 public:
  HvnaLayout(const VnaPreparation& preparation,
      std::span<const ProjectorAtomBlock> local, std::span<const ProjectorAtomBlock> halo,
      std::span<const double> energies);
  HvnaLayout(const HvnaLayout&) = delete;
  HvnaLayout& operator=(const HvnaLayout&) = delete;
  ProjectorArchiveLayout& local() noexcept { return local_; }
  ProjectorArchiveLayout& halo() noexcept { return halo_; }
  auto energies() noexcept { return projectors_.energies(); }
  std::span<int> lengths() noexcept { return projectors_.lengths(); }
 private:
  ProjectorArchiveLayout local_, halo_;
  HvnaProjectors projectors_;
};

} // namespace openmx::core
