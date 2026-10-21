#include "reciprocal_fixture.hpp"
#include "openmx_core_reciprocal_archive.h"
#include "allocation_faults.hpp"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
using namespace openmx::test::reciprocal;
using Archive=std::unique_ptr<OpenMXReciprocalArchive,decltype(&openmx_core_reciprocal_archive_destroy)>;
struct Temp {
  std::filesystem::path directory;
  Temp() {
    auto pattern=(std::filesystem::temp_directory_path()/"openmx-reciprocal-XXXXXX").string();
    auto *path=::mkdtemp(pattern.data());require(path,"cannot create archive test directory");directory=path;
  }
  ~Temp() { std::error_code error;std::filesystem::remove_all(directory,error); }
  std::string path(const char *name) const { return (directory/name).string(); }
};
OpenMXReciprocalArchiveInput input(Fixture& f,int kind) {
  return {kind,3,4,5,f.lmax,f.counts,f.n0,f.n1};
}
// Separate from the production flat accessors: original nested restart order.
std::vector<OpenMXReciprocalRow> reference(Fixture& f,int kind) {
  std::vector<OpenMXReciprocalRow> result;
  for (int s=0;s<3;++s) {
    if (kind==OPENMX_ARCHIVE_PAO)
      for (int l=0;l<=f.lmax[s];++l) for (int m=0;m<f.counts[s][l];++m)
        result.push_back({f.pao.transforms[s][l][m],4});
    if (kind==OPENMX_ARCHIVE_NEUTRAL_PROJECTOR)
      for (int l=0;l<3;++l) for (int m=0;m<2;++m)
        result.push_back({f.vps.neutral_transforms[s][l][m],5});
    if (kind==OPENMX_ARCHIVE_NEUTRAL_POTENTIAL) result.push_back({f.vps.potential_transform[s],5});
    if (kind==OPENMX_ARCHIVE_PRODUCT)
      for (int l1=0;l1<=f.lmax[s];++l1) for (int m1=0;m1<f.counts[s][l1];++m1)
        for (int l2=0;l2<=f.lmax[s];++l2) if (l1<=l2)
          for (int m2=0;m2<f.counts[s][l2];++m2) for (int l=0;l<=2*l2;++l)
            result.push_back({f.products.rows[s][l1][m1][l2][m2][l],5});
  }
  return result;
}
OpenMXReciprocalArchiveInput archive_input(Fixture& f,int kind) {
  // Nonlocal counts/spins differ from PAO basis multiplicities. Keep three
  // independent species, both spin states, and a species with no projectors.
  static const int projectors[]{2,0,3},spins[]{0,1,1};
  auto in=input(f,kind);in.projector_counts=projectors;in.projector_spin_lmax=spins;return in;
}
Archive create(OpenMXCoreContext *core,const OpenMXReciprocalArchiveInput& in,
    OpenMXReciprocalRows& rows) {
  char error[256];Archive result(openmx_core_reciprocal_archive_create(core,&in,&rows,error,sizeof(error)),
      openmx_core_reciprocal_archive_destroy);
  require(bool(result),error);return result;
}
std::vector<OpenMXReciprocalRow> expected(Fixture& f,int kind) {
  if (kind!=OPENMX_ARCHIVE_NONLOCAL) return reference(f,kind);
  const auto in=archive_input(f,kind);std::vector<OpenMXReciprocalRow> result;
  for (int s=0;s<3;++s) for (int spin=0;spin<=in.projector_spin_lmax[s];++spin)
    for (int p=1;p<=in.projector_counts[s];++p) result.push_back({f.vps.transforms[spin][s][p],4});
  return result;
}
void layout_test() {
  auto owner=context();Fixture f;f.allocate_all(owner.get());
  for (int kind=0;kind<5;++kind) {
    OpenMXReciprocalRows rows;auto archive=create(owner.get(),archive_input(f,kind),rows);
    const auto oracle=expected(f,kind);
    require(rows.species==3 && rows.rows==oracle.size() && rows.species_offsets[0]==0 &&
        rows.species_offsets[3]==rows.rows,"archive species/row count");
    for (std::size_t row=0;row<rows.rows;++row)
      require(rows.row[row].values==oracle[row].values && rows.row[row].count==oracle[row].count,
          "archive rows differ from legacy restart/MPI order");
    for (int s=0;s<3;++s) {
      auto in=archive_input(f,kind);in.species=s+1;OpenMXReciprocalRows prefix;
      auto partial=create(owner.get(),in,prefix);
      require(prefix.rows==rows.species_offsets[s+1],"archive species slice/empty species boundaries");
    }
  }
}
std::string contents(const std::string& path) {
  std::ifstream stream(path,std::ios::binary);
  return {std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
}
void io_test() {
  Temp temp;auto owner=context();Fixture f;f.allocate_all(owner.get());char error[256];
  for (int kind=0;kind<5;++kind) {
    OpenMXReciprocalRows rows;auto archive=create(owner.get(),archive_input(f,kind),rows);
    const auto oracle=expected(f,kind);const auto path=temp.path("table.bin");
    std::vector<double> frozen;
    for (std::size_t row=0;row<oracle.size();++row) for (int k=0;k<oracle[row].count;++k) {
      const double value=static_cast<double>(kind*10000+row*10+k)+.125;
      oracle[row].values[k]=value;frozen.push_back(value);
    }
    require(openmx_core_reciprocal_archive_write(archive.get(),path.c_str(),error,sizeof(error)),error);
    const auto raw=contents(path);
    require(raw.size()==frozen.size()*sizeof(double) &&
        !std::memcmp(raw.data(),frozen.data(),raw.size()),"archive binary format changed or padding serialized");
    for (const auto& row:oracle) for (int k=0;k<row.count;++k) row.values[k]=-9;
    require(openmx_core_reciprocal_archive_read(archive.get(),path.c_str(),error,sizeof(error)),error);
    std::size_t index=0;
    for (const auto& row:oracle) for (int k=0;k<row.count;++k)
      require(row.values[k]==frozen[index++],"archive round trip changed values/order");
    // Legacy readers accept extra bytes. They do not overwrite padded columns.
    { std::ofstream stream(path,std::ios::binary|std::ios::app);stream<<"trailing"; }
    require(openmx_core_reciprocal_archive_read(archive.get(),path.c_str(),error,sizeof(error)),error);
    if (kind==OPENMX_ARCHIVE_PAO) require(f.pao.transforms[0][0][0][4]==0,"archive overwrote PAO padding");
    if (kind==OPENMX_ARCHIVE_NONLOCAL) require(f.vps.transforms[0][0][0][0]==0,"archive overwrote reserved projector slot");
    if (kind==OPENMX_ARCHIVE_NEUTRAL_PROJECTOR) require(f.vps.neutral_transforms[0][0][0][5]==0,"archive overwrote neutral padding");
    // Preflight prevents truncated files from partially changing numeric rows.
    std::filesystem::resize_file(path,raw.size()-1);
    require(!openmx_core_reciprocal_archive_read(archive.get(),path.c_str(),error,sizeof(error)) && error[0],"short archive accepted");
    index=0;for (const auto& row:oracle) for (int k=0;k<row.count;++k)
      require(row.values[k]==frozen[index++],"short archive partially modified table");
  }
}
void failure_test() {
  auto owner=context();Fixture f;f.allocate_all(owner.get());OpenMXReciprocalRows rows{};char error[128];
  auto good=archive_input(f,OPENMX_ARCHIVE_PAO);auto archive=create(owner.get(),good,rows);const auto saved=rows;
  for (int variant=0;variant<11;++variant) {
    auto bad=good;
    switch (variant) {
      case 0:bad.kind=-1;break;case 1:bad.species=-1;break;case 2:bad.species=INT_MAX;break;
      case 3:bad.pao_points=INT_MAX;break;case 4:bad.maximum_angular=nullptr;break;
      case 5:bad.basis_counts=nullptr;break;case 6:bad.pao_points=0;break;
      case 7:bad.kind=OPENMX_ARCHIVE_NONLOCAL;bad.projector_counts=nullptr;break;
      case 8:bad.kind=OPENMX_ARCHIVE_NONLOCAL;bad.projector_spin_lmax=nullptr;break;
      case 9:bad.kind=OPENMX_ARCHIVE_PRODUCT;bad.neutral_points=INT_MAX;break;
      case 10:bad.kind=OPENMX_ARCHIVE_NEUTRAL_PROJECTOR;bad.neutral_points=INT_MAX;break;
    }
    auto *failed=openmx_core_reciprocal_archive_create(owner.get(),&bad,&rows,error,sizeof(error));
    require(!failed && error[0] && rows.row==saved.row && rows.rows==saved.rows,"bad archive metadata published state");
  }
  require(!openmx_core_reciprocal_archive_create(nullptr,&good,&rows,nullptr,0),"null context accepted");
  require(!openmx_core_reciprocal_archive_create(owner.get(),nullptr,&rows,nullptr,0),"null archive input accepted");
  require(!openmx_core_reciprocal_archive_create(owner.get(),&good,nullptr,nullptr,0),"null archive output accepted");
  require(!openmx_core_reciprocal_archive_read(nullptr,"unused",nullptr,0),"null archive read accepted");
  require(!openmx_core_reciprocal_archive_write(nullptr,"unused",nullptr,0),"null archive write accepted");
  Temp temp;const auto missing=temp.path("missing/table");
  for (int iteration=0;iteration<64;++iteration) {
    require(!openmx_core_reciprocal_archive_read(archive.get(),missing.c_str(),error,sizeof(error)),"missing file read accepted");
    require(!openmx_core_reciprocal_archive_write(archive.get(),missing.c_str(),error,sizeof(error)),"missing directory write accepted");
    require(!openmx_core_reciprocal_archive_write(archive.get(),nullptr,error,sizeof(error)),"missing path accepted");
    if (std::filesystem::exists("/dev/full"))
      require(!openmx_core_reciprocal_archive_write(archive.get(),"/dev/full",error,sizeof(error)),"buffered write failure ignored");
  }
  openmx_core_reciprocal_archive_destroy(nullptr);
}
void allocation_test() {
  auto owner=context();Fixture f;f.allocate_all(owner.get());char error[128];
  for (int kind=0;kind<5;++kind) {
    const auto in=archive_input(f,kind);OpenMXReciprocalRows rows{};
    auto archive=create(owner.get(),in,rows);const auto saved=rows;
    openmx::test::fail_allocation_after(10000);
    auto *test=openmx_core_reciprocal_archive_create(owner.get(),&in,&rows,error,sizeof(error));
    const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    require(test && count>1,"archive allocation interception unavailable");
    openmx_core_reciprocal_archive_destroy(test);rows=saved;
    for (long failure=0;failure<count;++failure) {
      const long live=openmx::test::live_allocations();
      openmx::test::fail_allocation_after(failure);
      test=openmx_core_reciprocal_archive_create(owner.get(),&in,&rows,error,sizeof(error));
      openmx::test::fail_allocation_after(-1);
      require(!test && error[0] && rows.row==saved.row && openmx::test::live_allocations()==live,
          "failed archive allocation leaked or changed output");
    }
  }
}
void lifetime_test() {
  auto owner=context();Archive archive(nullptr,openmx_core_reciprocal_archive_destroy);
  OpenMXReciprocalRows rows;
  {
    Fixture f;f.allocate_all(owner.get());archive=create(owner.get(),archive_input(f,OPENMX_ARCHIVE_PAO),rows);
    f.lmax[0]=-1;f.n0[0]=-1;
  }
  Temp temp;char error[128];
  for (std::size_t row=0;row<rows.rows;++row) for (int k=0;k<rows.row[row].count;++k)
    rows.row[row].values[k]=row+k;
  require(openmx_core_reciprocal_archive_write(archive.get(),temp.path("owned.bin").c_str(),error,sizeof(error)),error);
  // Destruction only releases plan metadata, never borrowed numeric rows.
  archive.reset();require(get(owner.get()).pao_transform.values[0]==0,"archive destruction freed calculation values");
}
void empty_test() {
  auto owner=context();Fixture f;OpenMXReciprocalRows rows;Temp temp;char error[128];
  for (int kind=0;kind<5;++kind) {
    auto in=archive_input(f,kind);in.species=0;
    auto archive=create(owner.get(),in,rows);
    require(rows.species==0 && rows.rows==0 && rows.species_offsets[0]==0,"empty archive is not empty");
    require(openmx_core_reciprocal_archive_write(archive.get(),temp.path("empty.bin").c_str(),error,sizeof(error)),error);
    require(std::filesystem::file_size(temp.path("empty.bin"))==0,"empty archive stored padding");
    require(openmx_core_reciprocal_archive_read(archive.get(),temp.path("empty.bin").c_str(),error,sizeof(error)),error);
  }
}
}
int main(int argc,char **argv) {
  try {
    const std::string name=argc==2?argv[1]:"";
    if (name=="layout") layout_test();else if (name=="io") io_test();else if (name=="failure") failure_test();
    else if (name=="allocation") allocation_test();else if (name=="lifetime") lifetime_test();else if (name=="empty") empty_test();
    else throw std::runtime_error("unknown reciprocal archive test");
    std::cout<<"PASS reciprocal archive "<<name<<'\n';return 0;
  } catch (const std::exception& e) {
    openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;
  }
}
