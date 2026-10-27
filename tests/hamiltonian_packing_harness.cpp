#include "openmx_hamiltonian_pack.h"
#include "openmx_core_native_resources.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>
#include "allocation_faults.hpp"
#ifdef PACKING_MPI_MODEL
#include <csetjmp>
static std::jmp_buf failure_target;
static bool catch_abort=false;
extern "C" void packed_test_abort(int status) {
    if(catch_abort)std::longjmp(failure_target,1);
    std::exit(status);
}
#endif

static void check(bool ok,const char *message) {
    if (!ok) { std::cerr << message << '\n'; MPI_Abort(MPI_COMM_WORLD,2); std::abort(); }
}
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
static Context calculation() {
    char error[256];
    Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
    check(bool(c),error); return c;
}

struct Geometry {
    static constexpr int atoms=5;
    std::array<int,6> species{{0,1,0,2,1,2}}, counts{{0,2,1,2,1,2}};
    std::array<int,3> orbitals{{1,3,2}};
    std::array<std::array<int,3>,6> neighbors{};
    std::array<int*,6> neighbor_rows{};
    std::vector<int> local{0};
    int rank=0,size=1;
    int owner(int atom) const { return (atom+1)%std::max(1,size-1); }
    explicit Geometry(MPI_Comm comm) {
        MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
        for(int a=1;a<=atoms;a++) {
            neighbors[a]={{a,a%atoms+1,(a+1)%atoms+1}};
            neighbor_rows[a]=neighbors[a].data();
        }
        // Deliberately use a non-global atom order and leave the last rank empty.
        for(int a=atoms;a>0;a--) if(owner(a)==rank)local.push_back(a);
    }
    OpenMXHamiltonianPacking input(int spin,int contracted,int reuse,int selected) {
        return {atoms,int(local.size()-1),spin,local.data(),species.data(),orbitals.data(),
                counts.data(),neighbor_rows.data(),selected,contracted,reuse,1};
    }
};

static double number(int seed,int family,int a,int h,int i,int j) {
    return seed*1000000+family*10000+a*1000+h*100+i*10+j;
}
// Stable RAII row adapters for the legacy sparse input boundary only.
struct Matrix {
    std::vector<std::vector<std::vector<std::vector<double>>>> values;
    std::vector<std::vector<std::vector<double*>>> rows;
    std::vector<std::vector<double**>> blocks;
    std::vector<double***> atoms;
    explicit Matrix(const Geometry &g) {
        size_t n=g.local.size();values.resize(n);rows.resize(n);blocks.resize(n);atoms.resize(n);
        for(size_t m=1;m<n;m++) {
            int a=g.local[m],nh=g.counts[a]+1,no=g.orbitals[g.species[a]];
            values[m].resize(nh);rows[m].resize(nh);blocks[m].resize(nh);
            for(int h=0;h<nh;h++) {
                int nj=g.orbitals[g.species[g.neighbors[a][h]]];
                values[m][h].resize(no);rows[m][h].resize(no);
                for(int i=0;i<no;i++) {values[m][h][i].resize(nj);rows[m][h][i]=values[m][h][i].data();}
                blocks[m][h]=rows[m][h].data();
            }
            atoms[m]=blocks[m].data();
        }
    }
    void fill(const Geometry &g,int seed,int family) {
        for(size_t m=1;m<values.size();m++) for(size_t h=0;h<values[m].size();h++)
            for(size_t i=0;i<values[m][h].size();i++) for(size_t j=0;j<values[m][h][i].size();j++)
                values[m][h][i][j]=number(seed,family,g.local[m],h,i,j);
    }
};
struct Matrices {
    std::vector<Matrix> matrices;
    std::array<double****,16> pointers{};
    OpenMXElectronicViews view{};
    explicit Matrices(const Geometry &g) {
        matrices.reserve(16);
        for(int k=0;k<16;k++) {matrices.emplace_back(g);pointers[k]=matrices.back().atoms.data();}
        view.OLP=pointers.data();view.H=pointers.data()+1;view.iHNL=pointers.data()+5;
        view.CntOLP=pointers.data()+8;view.CntH=pointers.data()+9;view.iCntHNL=pointers.data()+13;
    }
    void fill(const Geometry &g,int seed) {
        for(int k=0;k<16;k++)matrices[k].fill(g,seed+(k/8)*100,k%8);
    }
};

// Independent oracle: sort sparse coordinates by their required wire order,
// rather than copying the production nested packing loops or source pointers.
static std::vector<double> expected(const Geometry &g,int spin,int seed,int family) {
    using Record=std::tuple<int,int,int,int,int,double>;
    std::vector<Record> records;
    for(int a=1;a<=g.atoms;a++) for(int h=0;h<=g.counts[a];h++)
        for(int i=0;i<g.orbitals[g.species[a]];i++)
            for(int j=0;j<g.orbitals[g.species[g.neighbors[a][h]]];j++)
                records.emplace_back(g.owner(a),-a,spin==3?i:h,spin==3?h:i,j,number(seed,family,a,h,i,j));
    std::sort(records.begin(),records.end());std::vector<double> result;
    for(auto &r:records)result.push_back(std::get<5>(r));
    return result;
}
static void verify(OpenMXHamiltonianCache *c,const Geometry &g,
                   const OpenMXHamiltonianPacking &in,int s_seed,int h_seed,bool imaginary=true) {
    check(Set_Hamiltonian_GpuSolver_Packed_CacheReady(c),"cache not ready");
    check(Set_Hamiltonian_GpuSolver_Packed_OrderMode(c)==(in.spin_mode==3),"wrong order mode");
    auto reference=expected(g,in.spin_mode,s_seed,0);
    check(Set_Hamiltonian_GpuSolver_Packed_Size(c)==int(reference.size()),"wrong packed size");
    check(Set_Hamiltonian_GpuSolver_Packed_OwnsCache(c)==in.selected,"wrong ownership");
    auto *order=Set_Hamiltonian_GpuSolver_Packed_OrderGA(c);
    if(in.selected) {
        check(order && order[0]==0 && order[g.atoms+1]==0,"atom order boundaries");
        std::vector<std::pair<int,int>> atom_order;
        for(int a=1;a<=g.atoms;a++)atom_order.emplace_back(g.owner(a),-a);
        std::sort(atom_order.begin(),atom_order.end());
        for(int k=0;k<g.atoms;k++)check(order[k+1]==-atom_order[k].second,"rank/local atom order changed");
    } else check(!order,"non-owner holds atom order");
    for(int family=0;family<8;family++) {
        const double *v=family==0?Set_Hamiltonian_GpuSolver_Packed_Overlap(c):
            family<5?Set_Hamiltonian_GpuSolver_Packed_H(c,family-1):
                     Set_Hamiltonian_GpuSolver_Packed_ImNL(c,family-5);
        const int spins=in.spin_mode==3?4:in.spin_mode+1;
        bool present=in.selected && (family==0 || (family<=spins) || (family>=5 && in.spin_mode==3 && imaginary));
        check(bool(v)==present,"missing/extra component");
        if(present) {
            auto want=expected(g,in.spin_mode,family==0?s_seed:h_seed,family);
            check(std::equal(want.begin(),want.end(),v),"packed matrix differs from independent coordinate oracle");
        }
    }
    check(!Set_Hamiltonian_GpuSolver_Packed_H(c,-1) && !Set_Hamiltonian_GpuSolver_Packed_H(c,4),"H bounds");
    check(!Set_Hamiltonian_GpuSolver_Packed_ImNL(c,-1) && !Set_Hamiltonian_GpuSolver_Packed_ImNL(c,3),"ImNL bounds");
}

static int exercise(MPI_Comm comm) {
    Geometry g(comm);Matrices matrices(g);int cases=0;
    for(int spin:{0,1,3})for(int contracted:{0,1})for(int reuse:{0,1})for(int pattern:{0,1,2}) {
        auto a=calculation(),b=calculation();
        auto *ca=Set_Hamiltonian_Cache_Bind_Resources(a.get());
        auto *cb=Set_Hamiltonian_Cache_Bind_Resources(b.get());
        check(ca!=cb && ca==Set_Hamiltonian_Cache_Bind_Resources(a.get()),"calculation ownership/bind identity");
        check(!Set_Hamiltonian_GpuSolver_Packed_CacheReady(ca),"initial cache ready");
        int selected=pattern==0 || (pattern==1?g.rank==g.size-1:g.rank%2==1 || g.size==1);
        auto in=g.input(spin,contracted,reuse,selected);
        matrices.fill(g,2);
        check(Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view)==1,"initial rebuild");
        verify(ca,g,in,2+100*contracted,2+100*contracted);
        matrices.fill(g,7);
        check(Set_Hamiltonian_Build_GpuSolver_HS_Cache(cb,comm,&in,&matrices.view)==1,"second context rebuild");
        verify(ca,g,in,2+100*contracted,2+100*contracted);
        verify(cb,g,in,7+100*contracted,7+100*contracted);
        auto *old_s=Set_Hamiltonian_GpuSolver_Packed_Overlap(ca);
        matrices.fill(g,3);in.refresh_static=0;
        check(Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view)==!reuse,"reuse decision");
        verify(ca,g,in,(reuse?2:3)+100*contracted,3+100*contracted);
        if(reuse)check(old_s==Set_Hamiltonian_GpuSolver_Packed_Overlap(ca),"static allocation replaced");
        // Only one rank invalidates: all ranks must agree to rebuild before
        // entering optional collectives, even if that rank has no local atoms.
        if(g.rank==g.size-1)Set_Hamiltonian_Invalidate_GpuSolver_HS_Cache(ca);
        matrices.fill(g,4);
        check(Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view)==1,"collective invalidation");
        verify(ca,g,in,4+100*contracted,4+100*contracted);
        // Geometry/contracted refresh is explicit, even with unchanged sizes.
        matrices.fill(g,5);in.refresh_static=g.rank==0;
        check(Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view)==1,"explicit refresh");
        verify(ca,g,in,5+100*contracted,5+100*contracted);
        in.refresh_static=0;in.selected=g.rank==0 || !selected;
        check(Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view)==1 || g.size==1,"selected ranks changed");
        verify(ca,g,in,5+100*contracted,5+100*contracted);
        // Missing optional imaginary components must drop previous data.
        matrices.view.iHNL=nullptr;matrices.view.iCntHNL=nullptr;
        Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view);
        verify(ca,g,in,5+100*contracted,5+100*contracted,false);
        matrices.view.iHNL=matrices.pointers.data()+5;matrices.view.iCntHNL=matrices.pointers.data()+13;
        in.selected=0;
        check(!Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view),"no selected ranks");
        check(!Set_Hamiltonian_GpuSolver_Packed_CacheReady(ca) &&
              !Set_Hamiltonian_GpuSolver_Packed_Overlap(ca),"unselected cache not cleared");
        in.selected=1;Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view);
        openmx_core_native_resources_clear(a.get());
        ca=Set_Hamiltonian_Cache_Bind_Resources(a.get());
        check(!Set_Hamiltonian_GpuSolver_Packed_CacheReady(ca),"clear/rebind retained cache");
        in.selected=selected;verify(cb,g,in,7+100*contracted,7+100*contracted);
        Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view);
        openmx_core_finish(a.get());
        ca=Set_Hamiltonian_Cache_Bind_Resources(a.get());
        check(!Set_Hamiltonian_GpuSolver_Packed_CacheReady(ca),"finish/rebind retained cache");
        b.reset();
        Set_Hamiltonian_Build_GpuSolver_HS_Cache(ca,comm,&in,&matrices.view);
        verify(ca,g,in,5+100*contracted,5+100*contracted);
        cases++;
    }
    auto empty=calculation();auto *ce=Set_Hamiltonian_Cache_Bind_Resources(empty.get());
    auto in=g.input(3,0,1,g.rank==g.size-1);in.atom_count=0;in.local_atom_count=0;
    Set_Hamiltonian_Build_GpuSolver_HS_Cache(ce,comm,&in,&matrices.view);
    check(Set_Hamiltonian_GpuSolver_Packed_CacheReady(ce) &&
          Set_Hamiltonian_GpuSolver_Packed_Size(ce)==0 &&
          Set_Hamiltonian_GpuSolver_Packed_OwnsCache(ce)==in.selected,"globally empty layout");
    if(in.selected) {
        auto *order=Set_Hamiltonian_GpuSolver_Packed_OrderGA(ce);
        check(order && order[0]==0 && order[1]==0,"empty atom order sentinels");
    }
    return cases+1;
}
#ifdef PACKING_MPI_MODEL
static void allocation_failures() {
    Geometry g(MPI_COMM_WORLD);Matrices matrices(g);matrices.fill(g,8);
    auto survivor=calculation();auto *cb=Set_Hamiltonian_Cache_Bind_Resources(survivor.get());
    auto in=g.input(3,0,1,1);
    Set_Hamiltonian_Build_GpuSolver_HS_Cache(cb,MPI_COMM_WORLD,&in,&matrices.view);
    int failures=0;
    for(int fault=0;fault<64;fault++) {
        auto c=calculation();auto *cache=Set_Hamiltonian_Cache_Bind_Resources(c.get());
        // Production aborts MPI on allocation failure. The one-rank fixture
        // intercepts that fatal boundary solely to inspect partial ownership.
        catch_abort=true;
        if(setjmp(failure_target)==0) {
            openmx::test::fail_allocation_after(fault);
            Set_Hamiltonian_Build_GpuSolver_HS_Cache(cache,MPI_COMM_WORLD,&in,&matrices.view);
        } else failures++;
        openmx::test::fail_allocation_after(-1);catch_abort=false;
        openmx_core_native_resources_clear(c.get());
        cache=Set_Hamiltonian_Cache_Bind_Resources(c.get());
        check(!Set_Hamiltonian_GpuSolver_Packed_CacheReady(cache),"partial allocation survived clear");
        verify(cb,g,in,8,8);
    }
    check(failures>=8,"allocation fault injection did not reach packed buffers");
    std::cout << "PASS: " << failures << " injected fatal allocations drain through the calculation owner\n";
}
#endif

int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int cases=exercise(MPI_COMM_WORLD);MPI_Comm reversed,split;
    MPI_Comm_split(MPI_COMM_WORLD,0,-rank,&reversed);
    cases+=exercise(reversed);MPI_Comm_free(&reversed);
    MPI_Comm_split(MPI_COMM_WORLD,rank%2,rank,&split);
    cases+=exercise(split);MPI_Comm_free(&split);
    if(!rank)std::cout << "PASS: " << cases << " packed H/S scenarios; ownership, ordering, reuse, empty ranks, subcommunicators\n";
#ifdef PACKING_MPI_MODEL
    allocation_failures();
#endif
    MPI_Finalize();
}
