/**********************************************************************
  iterout_md.c:

     iterout_md.c is a subroutine to output calculated quantities
     at each MD step to filename.ene.

  Log of iterout_md.c:

     20/Sep./2007  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include "openmx_common.h"

void iterout_md(const OpenMXChemicalPotentials *chemical, const OpenMXDynamicsState *dynamics, const OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, double drctime, char fileSE[YOUSO10])
{
  int i,j,k;
  double dt,itermax,aa,dx,dy,dz,xido,yido,zido;
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */
  char fileEvsLC[YOUSO10];  

  if ((fp = fopen(fileSE,"a")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    /*  1 */      fprintf(fp,"%5d ",iter);
    /*  2 */      fprintf(fp,"%10.7f ",drctime);
    /*  3 */      fprintf(fp,"%10.7f ",energies->os[0]);
    /*  4 */      fprintf(fp,"%10.7f ",energies->is[0]);
    /*  5 */      fprintf(fp,"%10.7f ",energies->os[1]);
    /*  6 */      fprintf(fp,"%10.7f ",energies->band);
    /*  7 */      fprintf(fp,"%10.7f ",components->xc[0]);
    /*  8 */      fprintf(fp,"%10.7f ",components->xc[1]);
    /*  9 */      fprintf(fp,"%10.7f ",components->hartree[0]);
    /* 10 */      fprintf(fp,"%10.7f ",components->hartree[1]);
    /* 11 */      fprintf(fp,"%10.7f ",components->core);
    /* 12 */      fprintf(fp,"%10.7f ",Udc);
    /* 13 */      fprintf(fp,"%10.7f ",Ucoh);
    /* 14 */      fprintf(fp,"%10.7f ",thermo->ionic_kinetic);
    /* 15 */      fprintf(fp,"%10.7f ",thermo->total);
    /* 16 */      fprintf(fp,"%10.7f ",thermo->total+thermo->ionic_kinetic);
    /* 17 */      fprintf(fp,"%10.7f ",chemical->chemical_potential);
    /* 18 */      fprintf(fp,"%10.7f ",dynamics->target_temperature);
    /* 19 */      fprintf(fp,"%10.7f ",dynamics->temperature);
    /* 20 */      fprintf(fp,"%10.7f ",dynamics->nose_friction);
    /* 21 */      fprintf(fp,"%10.7f ",dynamics->nose_next_friction);
    /* 22 */      fprintf(fp,"%10.7f ",dynamics->nose_energy);
    /* 23 */      fprintf(fp,"%10.7f ",tv[1][1]*BohrR);
    /* 24 */      fprintf(fp,"%10.7f ",tv[1][2]*BohrR); 
    /* 25 */      fprintf(fp,"%10.7f ",tv[1][3]*BohrR);
    /* 26 */      fprintf(fp,"%10.7f ",tv[3][3]*BohrR);
    /* 27 */      fprintf(fp,"%10.7f ",Weight/Cell_Volume*10.0/6.022); 
    /* 28 */      fprintf(fp,"%10.7f ",GP*1.60219*100);
    /* 29 */      fprintf(fp,"%10.7f ",GT);
    /* 30 */      fprintf(fp,"%17.15f ",Max_Force);
    /* 31 */      fprintf(fp,"%10.7f ",Total_SpinS*2.0);
    /* 32 */      fprintf(fp,"%10.7f ",chemical->chemical_potential);
    /* 33 */      fprintf(fp,"%10.7f ", Gxyz[1][17]);
    /* 34 */      fprintf(fp,"%10.7f\n",Gxyz[1][17]);
    fclose(fp);
  }
  else
    printf("error in saving *.ene\n");

  /* E vs. LC */

  if (MD_switch==12){

    sprintf(fileEvsLC,"%s%s.EvsLC",filepath,filename);

    if ((fp = fopen(fileEvsLC,"a")) != NULL){

      fprintf(fp,"%15.10f ",tv[1][1]*BohrR);
      fprintf(fp,"%15.10f ",tv[1][2]*BohrR); 
      fprintf(fp,"%15.10f ",tv[1][3]*BohrR);
      fprintf(fp,"%15.10f ",tv[2][1]*BohrR);
      fprintf(fp,"%15.10f ",tv[2][2]*BohrR); 
      fprintf(fp,"%15.10f ",tv[2][3]*BohrR);
      fprintf(fp,"%15.10f ",tv[3][1]*BohrR);
      fprintf(fp,"%15.10f ",tv[3][2]*BohrR); 
      fprintf(fp,"%15.10f ",tv[3][3]*BohrR);
      fprintf(fp,"%15.10f\n",thermo->total);

      fclose(fp);
    }
    else
      printf("error in saving *.EvsLC\n");

  }  

  /* for calculation of the delta factor */

  if (MD_switch==16){

    sprintf(fileEvsLC,"%s%s.DF",filepath,filename);

    if ((fp = fopen(fileEvsLC,"a")) != NULL){

      fprintf(fp,"%18.12f ",Cell_Volume*BohrR*BohrR*BohrR/(double)atomnum);
      fprintf(fp,"%18.12f\n",thermo->total*eV2Hartree/(double)atomnum);

      fclose(fp);
    }
    else
      printf("error in saving *.EvsLC\n");

  }  

}


