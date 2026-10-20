/* Included after the frozen reference and the actual production functions. */
static int comparisons;
static void check_value(double value,double expected) {
  if (!isfinite(value) || !isfinite(expected) || fabs(value-expected)>2e-12*(1+fabs(expected))) {
    fprintf(stderr,"orbital comparison %d: %.17g != %.17g\n",comparisons,value,expected);
    abort();
  }
  ++comparisons;
}
static void compare_rows(double first[4][128],double second[4][128],int rows,int columns) {
  for (int r=0;r<rows;++r) for (int i=0;i<columns;++i) check_value(first[r][i],second[r][i]);
}
int main(void) {
  enum {S=2,L=5,M=3,N=24};
  double radii[S][N],orbitals[S][L][M][N];
  double *radii_rows[S],*orbital_rows[S][L][M],**orbital_angular[S][L],***orbital_species[S];
  int basis[S][L]={{2,2,2,1,1},{3,1,2,1,0}},contracted[S][L]={{1,1,1,1,1},{1,1,1,1,0}};
  int *basis_rows[S],*contracted_rows[S],mesh[S]={N,N-5},lmax[S]={4,3};
  int counts[S][128]={{0}},*count_rows[S],mapping[3]={0,1,2},species[3]={0,0,1};
  double coefficients[3][128][M]={{{0}}},*coefficient_rows[3][128],**coefficient_atoms[3];
  int empty_basis[S][L]={{0}},*empty_rows[S]={empty_basis[0],empty_basis[1]},empty_mesh[S]={6,6},empty_lmax[S]={0,0};
  for (int s=0;s<S;++s) {
    radii_rows[s]=radii[s];basis_rows[s]=basis[s];contracted_rows[s]=contracted[s];count_rows[s]=counts[s];
    orbital_species[s]=orbital_angular[s];
    for (int i=0;i<N;++i) radii[s][i]=exp(-3.4+s*.3+i*.23);
    int a=0;
    for (int l=0;l<L;++l) {
      orbital_angular[s][l]=orbital_rows[s][l];
      for (int m=0;m<M;++m) {
        orbital_rows[s][l][m]=orbitals[s][l][m];
        for (int i=0;i<N;++i) {
          double r=radii[s][i];orbitals[s][l][m][i]=(1+.2*s)*pow(r,l)*exp(-(.4+.17*m)*r)*(1+.03*r);
        }
      }
      for (int m=0;m<contracted[s][l];++m) for (int magnetic=0;magnetic<=2*l;++magnetic) {
        counts[s][a]=basis[s][l];
        for (int p=0;p<basis[s][l];++p) coefficients[s+1][a][p]=(.7+.1*p)/(1+.2*l);
        ++a;
      }
    }
  }
  for (int a=0;a<3;++a) {
    coefficient_atoms[a]=coefficient_rows[a];
    for (int i=0;i<128;++i) coefficient_rows[a][i]=coefficients[a][i];
  }
  F_M2G=mapping;WhatSpecies=species;CntCoes=coefficient_atoms;Spe_Specified_Num=count_rows;
  Spe_PAO_RV=radii_rows;Spe_PAO_RWF=orbital_species;
  OpenMXAtomicViews atomic={0};
  atomic.pao_radii=(OpenMXRadialView2){&radii[0][0],{S,N}};
  atomic.orbitals=(OpenMXRadialView4){&orbitals[0][0][0][0],{S,L,M,N}};
  atomic.species.pao_mesh=mesh;atomic.species.basis_lmax=lmax;atomic.species.basis=&basis[0][0];
  atomic.species.contracted_basis=&contracted[0][0];atomic.species.angular_channels=L;
  Set_Comp2Real();
  OpenMXOrbitalBuffers buffers;OpenMXOrbitalShape shape={4,M,1};char error[256];
  OpenMXOrbitalWorkspace *owner=openmx_core_orbital_create(&shape,&buffers,error,sizeof(error));
  if(!owner){fprintf(stderr,"%s\n",error);return 1;}
  for (int s=0;s<S;++s) {
    int no=0,nc=0;
    for(int l=0;l<=lmax[s];++l){no+=basis[s][l]*(2*l+1);nc+=contracted[s][l]*(2*l+1);}
    for(int point=0;point<40;++point) {
      double r;
      if(point==0)r=0;
      else if(point<5)r=radii[s][0]*(point*.17);
      else if(point<5+mesh[s])r=radii[s][point-5];
      else if(point==39)r=radii[s][mesh[s]-1]*1.1;
      else r=radii[s][0]+(radii[s][mesh[s]-1]-radii[s][0])*(point-5-mesh[s]+.37)/12.;
      for(int direction=0;direction<5;++direction) {
        double x=r*.3,y=r*.4,z=r*sqrt(.75);
        if(direction==1){x=-x;y=-y;z=-z;}
        if(direction==2){x=r;y=0;z=0;}
        if(direction==3){x=0;y=r;z=0;}
        if(direction==4){x=0;y=0;z=-r;}
        double old[7][4][128]={{{0}}},now[7][4][128]={{{0}}};
        double *rows[4];
        List_YOUSO[25]=4;List_YOUSO[24]=M;
        Spe_Num_Mesh_PAO=mesh;Spe_MaxL_Basis=lmax;Spe_Num_Basis=basis_rows;Spe_Num_CBasis=contracted_rows;
        reference_Get_Orbitals(s,x,y,z,old[0][0]);
        for(int i=0;i<4;++i)rows[i]=old[1][i];reference_Get_dOrbitals(s,x,y,z,rows);
        for(int i=0;i<4;++i)rows[i]=old[2][i];reference_Get_dOrbitals2(s,x,y,z,rows);
        reference_Get_Cnt_Orbitals(s+1,x,y,z,old[3][0]);
        for(int i=0;i<4;++i)rows[i]=old[4][i];reference_Get_Cnt_dOrbitals(s+1,x,y,z,rows);
        for(int i=0;i<4;++i)rows[i]=old[5][i];reference_Get_Cnt_dOrbitals2(s+1,x,y,z,rows);
        struct WORK_DORBITAL work;reference_Get_dOrbitals_init(&work);
        for(int i=0;i<4;++i)rows[i]=old[6][i];reference_Get_dOrbitals_work(s,x,y,z,rows,work);
        reference_Get_dOrbitals_free(work);
        /* Poison the former implicit dependencies. Only the passed PAO and
           species views may determine mesh lengths, counts and capacities. */
        Spe_Num_Mesh_PAO=empty_mesh;Spe_MaxL_Basis=empty_lmax;Spe_Num_Basis=empty_rows;Spe_Num_CBasis=empty_rows;
        List_YOUSO[25]=0;List_YOUSO[24]=1;
        Get_Orbitals(&atomic,s,x,y,z,now[0][0]);
        for(int i=0;i<4;++i)rows[i]=now[1][i];Get_dOrbitals(&atomic,s,x,y,z,rows);
        for(int i=0;i<4;++i)rows[i]=now[2][i];Get_dOrbitals2(&atomic,s,x,y,z,rows);
        Get_Cnt_Orbitals(&atomic,s+1,x,y,z,now[3][0]);
        for(int i=0;i<4;++i)rows[i]=now[4][i];Get_Cnt_dOrbitals(&atomic,s+1,x,y,z,rows);
        for(int i=0;i<4;++i)rows[i]=now[5][i];Get_Cnt_dOrbitals2(&atomic,s+1,x,y,z,rows);
        for(int i=0;i<4;++i)rows[i]=now[6][i];Get_dOrbitals_work(&atomic,s,x,y,z,rows,buffers);
        for(int k=0;k<7;++k)compare_rows(old[k],now[k],k==0 || k==3 ? 1 : 4,k>=3 && k<=5 ? nc : no);
      }
    }
    /* Cartesian finite differences provide a check independent of the
       frozen code. The force evaluator returns minus the spatial gradient. */
    /* The legacy generic g-orbital phi derivative omits a sin(theta)
       factor away from the equator. This ownership refactor preserves it;
       general g points above are regression checks against the frozen code.
       Test g gradients independently on the equator, and s/p/d/f everywhere. */
    for(int point=0;point<14;++point) {
      double xyz[3]={.35+(point%7)*.05,-.29,point<7 ? .41 : 0.};double values[4][128],plus[128],minus[128],*rows[4];
      for(int i=0;i<4;++i)rows[i]=values[i];Get_dOrbitals(&atomic,s,xyz[0],xyz[1],xyz[2],rows);
      for(int axis=0;axis<3;++axis) {
        double save=xyz[axis],h=1e-6;xyz[axis]=save+h;Get_Orbitals(&atomic,s,xyz[0],xyz[1],xyz[2],plus);
        xyz[axis]=save-h;Get_Orbitals(&atomic,s,xyz[0],xyz[1],xyz[2],minus);xyz[axis]=save;
        int checked=no;
        if(point<7 && lmax[s]>=4) checked-=basis[s][4]*9;
        for(int i=0;i<checked;++i) {
          double expected=-(plus[i]-minus[i])/(2*h),got=values[axis+1][i];
          if(!isfinite(got) || fabs(got-expected)>3e-6*(1+fabs(expected))) {
            fprintf(stderr,"orbital spatial derivative: species %d orbital %d component %d %.17g != %.17g\n",s,i,axis,got,expected);abort();
          }
          ++comparisons;
        }
      }
    }
  }
  openmx_core_orbital_destroy(owner);
  printf("PASS: %d orbital value/derivative comparisons, including poisoned globals and finite differences\n",comparisons);
  return 0;
}
