#include "force-grid-source.cpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace openmx::core::host;
template<int C> void tile(const double* dm,int columns,size_t pad,const float* orbitals,double* out,int first) {
  Vector acc[C];for(auto& v:acc)v=Vector(0.0);
  for(int j=0;j<columns;++j) {
    const Vector w{double(orbitals[j])};
    for(int c=0;c<C;++c)acc[c]+=w*load(dm+size_t(j)*pad+size_t(first+c)*lanes);
  }
  for(int c=0;c<C;++c)store(acc[c],out+size_t(first+c)*lanes);
}
template<int C> void candidate(const double* dm,int columns,size_t pad,const float* orbitals,double* out) {
  int v=0,n=pad/lanes;
  for(;v+C<=n;v+=C)tile<C>(dm,columns,pad,orbitals,out,v);
  if constexpr(C>4)if(v+4<=n){tile<4>(dm,columns,pad,orbitals,out,v);v+=4;}
  if constexpr(C>2)if(v+2<=n){tile<2>(dm,columns,pad,orbitals,out,v);v+=2;}
  for(;v<n;++v)tile<1>(dm,columns,pad,orbitals,out,v);
}
int main() {
  std::puts("rows,columns,method,seconds,bitwise_equal");
  for(int rows:{1,13,19,33})for(int columns:{0,1,4,13,19,33}) {
    size_t pad=force_padded(rows);
    std::vector<double> dm(pad*columns),out(pad),reference(pad);
    std::vector<float> orbital(columns);
    for(size_t i=0;i<dm.size();++i)dm[i]=std::cos(.01*i);
    for(int j=0;j<columns;++j)orbital[j]=std::sin(.13*j);
    density_rows(dm.data(),columns,pad,orbital.data(),reference.data());
    for(int method=0;method<5;++method) {
      auto run=[&]{switch(method){
        case 0:density_rows(dm.data(),columns,pad,orbital.data(),out.data());break;
        case 1:candidate<1>(dm.data(),columns,pad,orbital.data(),out.data());break;
        case 2:candidate<2>(dm.data(),columns,pad,orbital.data(),out.data());break;
        case 3:candidate<4>(dm.data(),columns,pad,orbital.data(),out.data());break;
        case 4:candidate<8>(dm.data(),columns,pad,orbital.data(),out.data());break;
      }};
      run();bool same=std::memcmp(out.data(),reference.data(),pad*sizeof(double))==0;
      if(!same)return 1;
      double t[5];for(int k=0;k<5;++k) {
        auto begin=std::chrono::steady_clock::now();for(int rep=0;rep<5000;++rep)run();
        t[k]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()/5000;
      }
      std::sort(t,t+5);std::printf("%d,%d,%d,%.12g,%d\n",rows,columns,method,t[2],same);
    }
  }
}
