// Local SVE256 register-tile study; every candidate is compared bitwise with
// the original scalar formulas, including output padding. Not application code.
#include "force-kernels-v2-source.cpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace openmx::core::host;

template<bool F,int T,bool L,int R,int V>
void tiled(const OpenMXForceVnaBlock& b,const float* const p[4],size_t pad) {
  const int outer=L?b.columns:b.rows,count=L?b.rows:b.columns;
  const int vectors=(count+lanes-1)/lanes;
  int v=0;
  for(;v+V<=vectors;v+=V) {
    int row=0;
    for(;row+R<=outer;row+=R)vna_tile<F,T,L,R,V>(b,p,pad,count,row,v);
    for(;row<outer;++row)vna_tile<F,T,L,1,V>(b,p,pad,count,row,v);
  }
  for(;v<vectors;++v) {
    int row=0;
    for(;row+R<=outer;row+=R)vna_tile<F,T,L,R,1>(b,p,pad,count,row,v);
    for(;row<outer;++row)vna_tile<F,T,L,1,1>(b,p,pad,count,row,v);
  }
}
template<bool F,int T,bool L>
void choose(int method,const OpenMXForceVnaBlock& b,const float* const p[4],size_t pad) {
  switch(method) {
    case 0:vna_rows<F,T,L>(b,p,pad);break;
    case 1:tiled<F,T,L,2,1>(b,p,pad);break;
    case 2:tiled<F,T,L,3,1>(b,p,pad);break;
    case 3:tiled<F,T,L,4,1>(b,p,pad);break;
    case 4:tiled<F,T,L,1,2>(b,p,pad);break;
    case 5:tiled<F,T,L,2,2>(b,p,pad);break;
    case 6:tiled<F,T,L,3,2>(b,p,pad);break;
    case 7:tiled<F,T,L,4,2>(b,p,pad);break;
    case 8:tiled<F,T,L,1,4>(b,p,pad);break;
    case 9:tiled<F,T,L,2,4>(b,p,pad);break;
  }
}
template<bool F,int T>
void choose(int method,const OpenMXForceVnaBlock& b,const float* const p[4],size_t pad) {
  if(b.lanes_rows)choose<F,T,true>(method,b,p,pad);
  else choose<F,T,false>(method,b,p,pad);
}
int main() {
  std::puts("rows,columns,projectors,fp32,terms,lanes_rows,method,seconds,bitwise_equal");
  for(int rows:{1,4,8,13,19})for(int columns:{1,4,8,13,19})
  for(int projectors:{0,7,972})for(int fp32:{0,1})for(int terms:{1,2})for(int lr:{0,1}) {
    std::array<std::vector<float>,8> data;
    std::array<std::vector<const float*>,8> pointers;
    for(int k=0;k<8;++k) {
      const int n=k<4?rows:columns;
      data[k].resize(size_t(n)*std::max(1,projectors));pointers[k].resize(n);
      for(size_t j=0;j<data[k].size();++j)data[k][j]=std::sin(.07*j+.31*k);
      for(int i=0;i<n;++i)pointers[k][i]=data[k].data()+size_t(i)*projectors;
    }
    const int count=lr?rows:columns,outer=lr?columns:rows;
    const size_t pad=force_padded(count),plane=size_t(projectors)*pad,stride=count+3;
    std::array<std::vector<double>,3> out,reference;
    for(auto& a:out)a.assign(size_t(outer)*stride+16,.1234567);
    OpenMXForceVnaBlock b{};b.rows=rows;b.columns=columns;b.projectors=projectors;
    b.fp32=fp32;b.terms=terms;b.lanes_rows=lr;b.output_stride=stride;
    for(int c=0;c<3;++c) {
      b.derivative[c]=pointers[c].data();b.column_derivative[c]=pointers[c+5].data();b.output[c]=out[c].data();
    }
    b.row_value=pointers[3].data();b.value=pointers[4].data();
    if(fp32){if(terms==1)vna_plain_rows<true,1>(b);else vna_plain_rows<true,2>(b);}
    else{if(terms==1)vna_plain_rows<false,1>(b);else vna_plain_rows<false,2>(b);}
    reference=out;
    std::vector<float> memory(std::max(size_t(1),4*plane));
    const float* planes[4];
    const int components=force_vna_planes(terms,lr);
    for(int c=0;c<components;++c) {
      const int source=lr?c:(c==0?4:c+4);
      transpose(pointers[source].data(),count,projectors,memory.data()+size_t(c)*plane,pad);
      planes[c]=memory.data()+size_t(c)*plane;
    }
    for(int method=0;method<10;++method) {
      auto run=[&] {
        if(fp32){if(terms==1)choose<true,1>(method,b,planes,pad);else choose<true,2>(method,b,planes,pad);}
        else{if(terms==1)choose<false,1>(method,b,planes,pad);else choose<false,2>(method,b,planes,pad);}
      };
      run();bool same=true;
      for(int c=0;c<3;++c)same=same && std::memcmp(out[c].data(),reference[c].data(),out[c].size()*sizeof(double))==0;
      if(!same){std::fprintf(stderr,"Mismatch %d %d %d %d %d %d %d\n",rows,columns,projectors,fp32,terms,lr,method);return 1;}
      double times[5];
      for(int t=0;t<5;++t) {
        auto start=std::chrono::steady_clock::now();
        for(int repeat=0;repeat<8;++repeat)run();
        times[t]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/8;
      }
      std::sort(times,times+5);
      std::printf("%d,%d,%d,%d,%d,%d,%d,%.12g,%d\n",rows,columns,projectors,fp32,terms,lr,method,times[2],same);
    }
  }
}
