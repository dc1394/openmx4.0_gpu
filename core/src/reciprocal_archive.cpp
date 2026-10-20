#include <openmx/core/reciprocal_archive.hpp>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>

namespace openmx::core {
namespace {
struct CloseFile {
  void operator()(std::FILE *stream) const noexcept { std::fclose(stream); }
};
using File=std::unique_ptr<std::FILE,CloseFile>;
File file(const char *path,const char *mode) {
  if (!path || !*path) throw std::invalid_argument("missing reciprocal archive path");
  File result(std::fopen(path,mode));
  if (!result) throw std::runtime_error("cannot open reciprocal archive");
  return result;
}
void require(bool condition) {
  if (!condition) throw std::invalid_argument("reciprocal archive metadata exceeds owned table dimensions");
}
}

ReciprocalArchive::ReciprocalArchive(const OpenMXReciprocalViews& v,
    const OpenMXReciprocalArchiveInput& in)
{
  require(in.species>=0 && in.kind>=OPENMX_ARCHIVE_PAO && in.kind<=OPENMX_ARCHIVE_PRODUCT);
  const bool uniform=in.kind==OPENMX_ARCHIVE_PAO || in.kind==OPENMX_ARCHIVE_NONLOCAL;
  const int points=uniform?in.pao_points:in.neutral_points;
  require(points>0);
  // Reject impossible species counts before iterating any caller metadata.
  const std::size_t capacity=in.kind==OPENMX_ARCHIVE_PAO?v.pao_transform.extent[0]:
      in.kind==OPENMX_ARCHIVE_NONLOCAL?v.nonlocal_transform.extent[1]:
      in.kind==OPENMX_ARCHIVE_NEUTRAL_PROJECTOR?v.neutral_transform.extent[0]:
      in.kind==OPENMX_ARCHIVE_NEUTRAL_POTENTIAL?v.potential_transform.extent[0]:v.product_transform.species;
  require(static_cast<std::size_t>(in.species)<=capacity);
  for (int s=0;s<in.species;++s) {
    switch (in.kind) {
      case OPENMX_ARCHIVE_PAO:
        require(v.pao_transform.values && in.maximum_angular && in.basis_counts && in.basis_counts[s]);
        require(in.maximum_angular[s]>=0 && static_cast<std::size_t>(in.maximum_angular[s])<v.pao_transform.extent[1] &&
            static_cast<std::size_t>(points)<=v.pao_transform.extent[3]);
        for (int l=0;l<=in.maximum_angular[s];++l) {
          const int count=in.basis_counts[s][l];
          require(count>=0 && static_cast<std::size_t>(count)<=v.pao_transform.extent[2]);
          for (int m=0;m<count;++m) append(openmx_radial_row4(&v.pao_transform,s,l,m),points);
        }
        break;
      case OPENMX_ARCHIVE_NONLOCAL:
        require(v.nonlocal_transform.values && in.projector_counts && in.projector_spin_lmax);
        require(in.projector_spin_lmax[s]>=0 && static_cast<std::size_t>(in.projector_spin_lmax[s])<v.nonlocal_transform.extent[0] &&
            in.projector_counts[s]>=0 && static_cast<std::size_t>(in.projector_counts[s])<v.nonlocal_transform.extent[2] &&
            static_cast<std::size_t>(points)<=v.nonlocal_transform.extent[3]);
        for (int spin=0;spin<=in.projector_spin_lmax[s];++spin)
          for (int p=1;p<=in.projector_counts[s];++p)
            append(openmx_radial_row4(&v.nonlocal_transform,spin,s,p),points);
        break;
      case OPENMX_ARCHIVE_NEUTRAL_PROJECTOR:
        require(v.neutral_transform.values && static_cast<std::size_t>(points)+2<=v.neutral_transform.extent[3]);
        for (std::size_t l=0;l<v.neutral_transform.extent[1];++l)
          for (std::size_t m=0;m<v.neutral_transform.extent[2];++m)
            append(openmx_radial_row4(&v.neutral_transform,s,l,m),points);
        break;
      case OPENMX_ARCHIVE_NEUTRAL_POTENTIAL:
        require(v.potential_transform.values && static_cast<std::size_t>(points)+2<=v.potential_transform.extent[1]);
        append(openmx_radial_row2(&v.potential_transform,s),points);
        break;
      case OPENMX_ARCHIVE_PRODUCT: {
        const auto& product=v.product_transform;
        require(product.layout && static_cast<std::size_t>(points)+2<=product.stride);
        const std::size_t first=product.layout[s].first_row;
        const std::size_t last=static_cast<std::size_t>(s)+1<product.species?
            product.layout[s+1].first_row:product.rows;
        require(first<=last && last<=product.rows && (first==last || product.values));
        for (std::size_t row=first;row<last;++row) append(product.values+row*product.stride,points);
        break;
      }
    }
    offsets_.push_back(rows_.size());
  }
}

void ReciprocalArchive::append(double *values,int count) {
  if (static_cast<std::size_t>(count)>std::numeric_limits<std::size_t>::max()/sizeof(double))
    throw std::length_error("reciprocal archive row size overflows size_t");
  const auto bytes=static_cast<std::size_t>(count)*sizeof(double);
  if (bytes>std::numeric_limits<std::size_t>::max()-bytes_)
    throw std::length_error("reciprocal archive byte count overflows size_t");
  rows_.push_back({values,count});bytes_+=bytes;
}
OpenMXReciprocalRows ReciprocalArchive::rows() const noexcept {
  return {offsets_.size()-1,rows_.size(),offsets_.data(),rows_.data()};
}
void ReciprocalArchive::read(const char *path) {
  auto input=file(path,"rb");
  // No second copy of potentially large orbital-product data. Reject short
  // restart files before touching numeric storage, then stream into each row.
  if (std::fseek(input.get(),0,SEEK_END)) throw std::runtime_error("cannot seek reciprocal archive");
  const long size=std::ftell(input.get());
  if (size<0 || static_cast<unsigned long>(size)<bytes_)
    throw std::runtime_error("truncated reciprocal archive");
  if (std::fseek(input.get(),0,SEEK_SET)) throw std::runtime_error("cannot rewind reciprocal archive");
  for (const auto& row:rows_)
    if (std::fread(row.values,sizeof(double),row.count,input.get())!=static_cast<std::size_t>(row.count))
      throw std::runtime_error("failed to read reciprocal archive row");
}
void ReciprocalArchive::write(const char *path) const {
  auto output=file(path,"wb");
  for (const auto& row:rows_)
    if (std::fwrite(row.values,sizeof(double),row.count,output.get())!=static_cast<std::size_t>(row.count))
      throw std::runtime_error("failed to write reciprocal archive row");
  // fclose reports buffered write errors; ownership is already released so
  // the error path cannot close the same stream twice.
  if (std::fclose(output.release())) throw std::runtime_error("failed to close reciprocal archive");
}
} // namespace openmx::core

struct OpenMXReciprocalArchive {
  OpenMXReciprocalArchive(const OpenMXReciprocalViews& views,const OpenMXReciprocalArchiveInput& input)
      : implementation(views,input) {}
  openmx::core::ReciprocalArchive implementation;
};
namespace {
template<class Function> int checked(char *error,std::size_t capacity,Function&& function) noexcept {
  if (error && capacity) *error='\0';
  try { function();return 1; }
  catch (const std::exception& e) { if (error && capacity) std::snprintf(error,capacity,"%s",e.what()); }
  catch (...) { if (error && capacity) std::snprintf(error,capacity,"unexpected reciprocal archive exception"); }
  return 0;
}
}
extern "C" OpenMXReciprocalArchive *openmx_core_reciprocal_archive_create(OpenMXCoreContext *context,
    const OpenMXReciprocalArchiveInput *input,OpenMXReciprocalRows *rows,char *error,std::size_t capacity) {
  OpenMXReciprocalArchive *result=nullptr;
  checked(error,capacity,[&] {
    if (!input || !rows) throw std::invalid_argument("missing reciprocal archive input or output");
    OpenMXReciprocalViews views;
    char reason[128];
    if (!openmx_core_reciprocal_get(context,&views,reason,sizeof(reason))) throw std::invalid_argument(reason);
    auto owner=std::make_unique<OpenMXReciprocalArchive>(views,*input);
    *rows=owner->implementation.rows();result=owner.release();
  });return result;
}
extern "C" int openmx_core_reciprocal_archive_read(OpenMXReciprocalArchive *archive,
    const char *path,char *error,std::size_t capacity) {
  return checked(error,capacity,[&] {
    if (!archive) throw std::invalid_argument("missing reciprocal archive");
    archive->implementation.read(path);
  });
}
extern "C" int openmx_core_reciprocal_archive_write(const OpenMXReciprocalArchive *archive,
    const char *path,char *error,std::size_t capacity) {
  return checked(error,capacity,[&] {
    if (!archive) throw std::invalid_argument("missing reciprocal archive");
    archive->implementation.write(path);
  });
}
extern "C" void openmx_core_reciprocal_archive_destroy(OpenMXReciprocalArchive *archive) { delete archive; }
