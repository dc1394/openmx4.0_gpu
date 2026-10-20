/* Included after the frozen reference and the actual production functions. */
#include <complex.h>
static int comparisons;
static int derivative_comparisons;
static double max_derivative_error;
static double max_g_finite_difference_error,max_g_polynomial_error;
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
static void check_gradient(const char *kind,int orbital,int axis,double actual,double expected,double tolerance) {
  double error=fabs(actual-expected);
  if (!isfinite(actual) || !isfinite(expected) || error>tolerance*(1+fabs(expected))) {
    fprintf(stderr,"%s: orbital %d component %d %.17g != %.17g\n",kind,orbital,axis,actual,expected);
    abort();
  }
  if(error>max_derivative_error) max_derivative_error=error;
  ++comparisons;
  ++derivative_comparisons;
}
/* Independent Cartesian polynomials for the nine real, normalized g
   harmonics. Complex-step differentiation avoids a subtractive finite
   difference and does not use the production Legendre recurrence. */
static void g_polynomials(long double complex x,long double complex y,long double complex z,
                          long double complex values[9]) {
  const long double pi=acosl(-1.L);
  long double complex xx=x*x,yy=y*y,zz=z*z,r2=xx+yy+zz,r4=r2*r2;
  values[0]=3.L/(16.L*sqrtl(pi))*(35.L*zz*zz-30.L*zz*r2+3.L*r4)/r4;
  values[1]=3.L/8.L*sqrtl(10.L/pi)*x*z*(7.L*zz-3.L*r2)/r4;
  values[2]=3.L/8.L*sqrtl(10.L/pi)*y*z*(7.L*zz-3.L*r2)/r4;
  values[3]=3.L/8.L*sqrtl(5.L/pi)*(xx-yy)*(7.L*zz-r2)/r4;
  values[4]=3.L/4.L*sqrtl(5.L/pi)*x*y*(7.L*zz-r2)/r4;
  values[5]=3.L/8.L*sqrtl(70.L/pi)*x*z*(xx-3.L*yy)/r4;
  values[6]=3.L/8.L*sqrtl(70.L/pi)*y*z*(3.L*xx-yy)/r4;
  values[7]=3.L/16.L*sqrtl(35.L/pi)*(xx*xx-6.L*xx*yy+yy*yy)/r4;
  values[8]=3.L/4.L*sqrtl(35.L/pi)*x*y*(xx-yy)/r4;
}
static const double positions[][3]={
  {.35,-.29,.41},{-.35,.29,-.41},{.35,.29,.41},{-.35,-.29,-.41},
  {.57,.18,0},{-.57,.18,0},{0,.61,0},{0,-.61,0},{.61,0,0},{-.61,0,0},
  {0,0,.61},{0,0,-.61},{.001,-.002,.61},{-.001,.002,-.61},
  {1e-8,-2e-8,.61},{-1e-8,2e-8,-.61},{1e-12,2e-12,.61},{1e-12,2e-12,-.61},
  {-1e-14,2e-14,.61},{-1e-14,2e-14,-.61},{.7,-.4,.2},{.2,.3,-.8},
  {.001,.002,.003}
};
int main(void) {
  enum {S=2,L=5,M=3,N=24};
  double radii[S][N],orbitals[S][L][M][N];
  double *radii_rows[S],*orbital_rows[S][L][M],**orbital_angular[S][L],***orbital_species[S];
  int basis[S][L]={{2,2,2,1,2},{3,1,2,1,0}},contracted[S][L]={{1,1,1,1,1},{1,1,1,1,0}};
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
        for(int k=0;k<7;++k) {
          const int cnt=k>=3 && k<=5,columns=cnt ? nc : no;
          const int lower=columns-(lmax[s]>=4 ? 9*(cnt ? contracted[s][4] : basis[s][4]) : 0);
          compare_rows(old[k],now[k],1,columns);
          if(k!=0 && k!=3)
            for(int row=1;row<4;++row) for(int i=0;i<lower;++i) check_value(now[k][row][i],old[k][row][i]);
        }
        for(int i=0;i<no;++i) for(int axis=1;axis<4;++axis) {
          check_value(now[2][axis][i],-now[1][axis][i]);
          check_value(now[6][axis][i],now[1][axis][i]);
        }
        for(int i=0;i<nc;++i) for(int axis=1;axis<4;++axis)
          check_value(now[5][axis][i],-now[4][axis][i]);
        double inlined[4][128]={{0}},*inline_rows[4],grid[128];
        for(int i=0;i<4;++i)inline_rows[i]=inlined[i];
        force_cuda(&atomic,s,x,y,z,inline_rows,buffers);
        for(int axis=1;axis<4;++axis)for(int i=0;i<no;++i)check_value(inlined[axis][i],now[1][axis][i]);
        force_hip(&atomic,s,x,y,z,inline_rows,buffers);
        for(int axis=1;axis<4;++axis)for(int i=0;i<no;++i)check_value(inlined[axis][i],now[1][axis][i]);
        grid_cuda_0(&atomic,s,x,y,z,grid);for(int i=0;i<no;++i)check_value(grid[i],now[0][0][i]);
        grid_cuda_1(&atomic,s,x,y,z,grid);for(int i=0;i<no;++i)check_value(grid[i],now[0][0][i]);
        grid_hip_0(&atomic,s,x,y,z,grid);for(int i=0;i<no;++i)check_value(grid[i],now[0][0][i]);
        grid_hip_1(&atomic,s,x,y,z,grid);for(int i=0;i<no;++i)check_value(grid[i],now[0][0][i]);
      }
    }
    /* Cartesian finite differences provide a check independent of the
       frozen code. The force evaluator returns minus the spatial gradient. */
    for(size_t point=0;point<sizeof(positions)/sizeof(positions[0]);++point) {
      double xyz[3]={positions[point][0],positions[point][1],positions[point][2]};
      /* Non-g bases keep the legacy acos coordinate conversion. Its
         roundoff near (but not on) the axis makes value differences noisy.
         Every near-axis point is checked for the g-containing species. */
      if(lmax[s]<4 && hypot(xyz[0],xyz[1])>0 && hypot(xyz[0],xyz[1])<1e-6) continue;
      double values[5][4][128],shifted[2][4][128],*rows[4];
      for(int i=0;i<4;++i)rows[i]=values[0][i];Get_dOrbitals(&atomic,s,xyz[0],xyz[1],xyz[2],rows);
      for(int i=0;i<4;++i)rows[i]=values[1][i];Get_dOrbitals2(&atomic,s,xyz[0],xyz[1],xyz[2],rows);
      for(int i=0;i<4;++i)rows[i]=values[2][i];Get_dOrbitals_work(&atomic,s,xyz[0],xyz[1],xyz[2],rows,buffers);
      for(int i=0;i<4;++i)rows[i]=values[3][i];Get_Cnt_dOrbitals(&atomic,s+1,xyz[0],xyz[1],xyz[2],rows);
      for(int i=0;i<4;++i)rows[i]=values[4][i];Get_Cnt_dOrbitals2(&atomic,s+1,xyz[0],xyz[1],xyz[2],rows);
      for(int axis=0;axis<3;++axis) {
        const double save=xyz[axis],h=1e-5,offsets[4]={-2,-1,1,2};
        for(int sample=0;sample<4;++sample) {
          xyz[axis]=save+offsets[sample]*h;
          Get_Orbitals(&atomic,s,xyz[0],xyz[1],xyz[2],shifted[0][sample]);
          Get_Cnt_Orbitals(&atomic,s+1,xyz[0],xyz[1],xyz[2],shifted[1][sample]);
        }
        xyz[axis]=save;
        for(int variant=0;variant<5;++variant) {
          const int cnt=variant>=3,columns=cnt ? nc : no,sign=(variant==1 || variant==4) ? 1 : -1;
          for(int i=0;i<columns;++i) {
            double expected=sign*(shifted[cnt][0][i]-8*shifted[cnt][1][i]+8*shifted[cnt][2][i]-shifted[cnt][3][i])/(12*h);
            /* The l<=3 species retains its historical acos coordinates;
               the independent polynomial test below checks g poles tightly. */
            check_gradient("spatial finite difference",i,axis,values[variant][axis+1][i],expected,s ? 3e-6 : 2e-8);
            if(s==0 && i>=(cnt ? 16 : 25))
              max_g_finite_difference_error=fmax(max_g_finite_difference_error,fabs(values[variant][axis+1][i]-expected));
          }
        }
      }
    }
  }
  /* A known reproduction must differ from the old, buggy derivative. */
  {
    double old[4][128],now[4][128],*rows[4];
    List_YOUSO[25]=4;List_YOUSO[24]=M;
    Spe_Num_Mesh_PAO=mesh;Spe_MaxL_Basis=lmax;Spe_Num_Basis=basis_rows;Spe_Num_CBasis=contracted_rows;
    for(int i=0;i<4;++i)rows[i]=old[i];reference_Get_dOrbitals(0,.35,-.29,.41,rows);
    for(int i=0;i<4;++i)rows[i]=now[i];Get_dOrbitals(&atomic,0,.35,-.29,.41,rows);
    if(fabs(old[1][26]-now[1][26])<1e-4)abort();
    printf("g derivative reproduction: old %.17g, corrected %.17g\n",old[1][26],now[1][26]);
  }
  /* With a constant radial function the full evaluator is the normalized
     Cartesian harmonic. Check every g component and both pole limits. */
  for(int mul=0;mul<M;++mul)for(int i=0;i<N;++i)orbitals[0][4][mul][i]=1+.2*mul;
  for(size_t point=0;point+1<sizeof(positions)/sizeof(positions[0]);++point) {
    const double *xyz=positions[point];
    double values[7][4][128],*rows[4],plain[128],contracted_values[128];
    for(int i=0;i<4;++i)rows[i]=values[0][i];Get_dOrbitals(&atomic,0,xyz[0],xyz[1],xyz[2],rows);
    for(int i=0;i<4;++i)rows[i]=values[1][i];Get_dOrbitals2(&atomic,0,xyz[0],xyz[1],xyz[2],rows);
    for(int i=0;i<4;++i)rows[i]=values[2][i];Get_dOrbitals_work(&atomic,0,xyz[0],xyz[1],xyz[2],rows,buffers);
    for(int i=0;i<4;++i)rows[i]=values[3][i];Get_Cnt_dOrbitals(&atomic,1,xyz[0],xyz[1],xyz[2],rows);
    for(int i=0;i<4;++i)rows[i]=values[4][i];Get_Cnt_dOrbitals2(&atomic,1,xyz[0],xyz[1],xyz[2],rows);
    for(int i=0;i<4;++i)rows[i]=values[5][i];force_cuda(&atomic,0,xyz[0],xyz[1],xyz[2],rows,buffers);
    for(int i=0;i<4;++i)rows[i]=values[6][i];force_hip(&atomic,0,xyz[0],xyz[1],xyz[2],rows,buffers);
    Get_Orbitals(&atomic,0,xyz[0],xyz[1],xyz[2],plain);
    Get_Cnt_Orbitals(&atomic,1,xyz[0],xyz[1],xyz[2],contracted_values);
    long double complex oracle[9],shifted[9];
    g_polynomials(xyz[0],xyz[1],xyz[2],oracle);
    for(int axis=0;axis<3;++axis) {
      const long double h=1e-24L;
      long double complex shifted_xyz[3]={xyz[0],xyz[1],xyz[2]};
      shifted_xyz[axis]+=h*I;
      g_polynomials(shifted_xyz[0],shifted_xyz[1],shifted_xyz[2],shifted);
      for(int magnetic=0;magnetic<9;++magnetic) {
        const int cnt_index=16+magnetic;
        const double phase=magnetic==0 || magnetic%2 ? -1.0 : 1.0;
        double coefficient=0;
        for(int mul=0;mul<basis[0][4];++mul)coefficient+=coefficients[1][cnt_index][mul]*(1+.2*mul);
        check_value(contracted_values[cnt_index],phase*coefficient*creall(oracle[magnetic]));
        for(int variant=0;variant<7;++variant) {
          const int cnt=variant==3 || variant==4,sign=variant==1 || variant==4 ? 1 : -1;
          for(int mul=0;mul<(cnt ? 1 : basis[0][4]);++mul) {
            const int index=cnt ? cnt_index : 25+9*mul+magnetic;
            const double factor=cnt ? coefficient*phase : 1+.2*mul;
            const double expected=sign*factor*cimagl(shifted[magnetic])/h;
            check_gradient("Cartesian polynomial",index,axis,values[variant][axis+1][index],expected,2e-11);
            max_g_polynomial_error=fmax(max_g_polynomial_error,fabs(values[variant][axis+1][index]-expected));
            if(variant<5)check_value(values[variant][0][index],factor*creall(oracle[magnetic]));
            if(!cnt)check_value(plain[index],factor*creall(oracle[magnetic]));
          }
        }
      }
    }
  }
  openmx_core_orbital_destroy(owner);
  printf("PASS: %d orbital comparisons; %d independent derivatives, maximum absolute error %.3g; "
         "all g components, general angles, both poles, contracted/reversed/reused and CUDA/HIP Force/grid fallbacks\n",
         comparisons,derivative_comparisons,max_derivative_error);
  printf("g-only maximum errors: finite differences %.3g; Cartesian polynomial derivatives %.3g\n",
         max_g_finite_difference_error,max_g_polynomial_error);
  return 0;
}
