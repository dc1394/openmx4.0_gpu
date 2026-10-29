#include <openmx/basis/vps_file.hpp>
#include <iostream>

int main(int argc,char** argv)
{
  try {
    if(argc!=2){std::cerr<<"usage: vps_example FILE.vps\n";return 1;}
    const auto file=openmx::basis::VpsFile::from_file(argv[1]);
    std::cout<<"AtomSpecies="<<file.metadata().atom_species<<" projectors="<<file.angular().size()
             <<" spin channels="<<file.metadata().spin_channels()<<'\n';
    if(!file.angular().empty())std::cout<<"projector(0,0,r=0.85 bohr)="<<file.sample_projector(0,0,.85)<<'\n';
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
