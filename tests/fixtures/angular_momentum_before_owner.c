/* Frozen Generate_L_Matrix from aeadecf0; only the function name changed. */
static void legacy_Generate_L_Matrix(int Lmax, double ***Lx0, double ***Ly0, double ***Lz0)
{
  int L,M1,M2,M3,M,m1,m2;
  dcomplex ***CLx,***CLy,***CLz;
  dcomplex ***CLx1,***CLy1,***CLz1;
  dcomplex Ctmp1,Ctmpx,Ctmpy,Ctmpz;

  /* allocation of arrays */

  CLx = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1));
  for (L=0; L<=Lmax; L++){
    CLx[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1));
    for (M=0; M<=(2*L); M++){
      CLx[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1));
      for (M1=0; M1<=(2*L); M1++)  CLx[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLy = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1));
  for (L=0; L<=Lmax; L++){
    CLy[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1));
    for (M=0; M<=(2*L); M++){
      CLy[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1));
      for (M1=0; M1<=(2*L); M1++)  CLy[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLz = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1));
  for (L=0; L<=Lmax; L++){
    CLz[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1));
    for (M=0; M<=(2*L); M++){
      CLz[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1));
      for (M1=0; M1<=(2*L); M1++)  CLz[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLx1 = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1));
  for (L=0; L<=Lmax; L++){
    CLx1[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1));
    for (M=0; M<=(2*L); M++){
      CLx1[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1));
      for (M1=0; M1<=(2*L); M1++)  CLx1[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLy1 = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1));
  for (L=0; L<=Lmax; L++){
    CLy1[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1));
    for (M=0; M<=(2*L); M++){
      CLy1[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1));
      for (M1=0; M1<=(2*L); M1++)  CLy1[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLz1 = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1));
  for (L=0; L<=Lmax; L++){
    CLz1[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1));
    for (M=0; M<=(2*L); M++){
      CLz1[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1));
      for (M1=0; M1<=(2*L); M1++)  CLz1[L][M][M1] = Complex(0.0,0.0);
    }
  }

  /* set complex matrices */

  for (L=0; L<=Lmax; L++){

    for (M1=0; M1<=2*L; M1++){
      for (M2=0; M2<=2*L; M2++){

        m1 = M1 - L;
        m2 = M2 - L;

        if (M1==(M2+1)) CLx[L][M1][M2].r  = 0.5*sqrt((L-m2)*(L+m2+1.0));
        if (M1==(M2-1)) CLx[L][M1][M2].r += 0.5*sqrt((L+m2)*(L-m2+1.0));

        if (M1==(M2+1)) CLy[L][M1][M2].i  =-0.5*sqrt((L-m2)*(L+m2+1.0));
        if (M1==(M2-1)) CLy[L][M1][M2].i -=-0.5*sqrt((L+m2)*(L-m2+1.0));

        if (M1==M2)     CLz[L][M1][M2].r  = m1;
      }
    }
  }

  /* transformation */

  for (L=0; L<=Lmax; L++){

    for (M1=0; M1<=2*L; M1++){
      for (M2=0; M2<=2*L; M2++){

        Ctmpx = Complex(0.0,0.0);
        Ctmpy = Complex(0.0,0.0);
        Ctmpz = Complex(0.0,0.0);

        for (M3=0; M3<=2*L; M3++){

          Ctmp1 = Comp2Real[L][M2][M3];

          Ctmpx.r += CLx[L][M1][M3].r*Ctmp1.r - CLx[L][M1][M3].i*Ctmp1.i;
          Ctmpx.i += CLx[L][M1][M3].r*Ctmp1.i + CLx[L][M1][M3].i*Ctmp1.r;

          Ctmpy.r += CLy[L][M1][M3].r*Ctmp1.r - CLy[L][M1][M3].i*Ctmp1.i;
          Ctmpy.i += CLy[L][M1][M3].r*Ctmp1.i + CLy[L][M1][M3].i*Ctmp1.r;

          Ctmpz.r += CLz[L][M1][M3].r*Ctmp1.r - CLz[L][M1][M3].i*Ctmp1.i;
          Ctmpz.i += CLz[L][M1][M3].r*Ctmp1.i + CLz[L][M1][M3].i*Ctmp1.r;

	}

        CLx1[L][M1][M2] = Ctmpx;
        CLy1[L][M1][M2] = Ctmpy;
        CLz1[L][M1][M2] = Ctmpz;
      }
    }
  }

  for (L=0; L<=Lmax; L++){

    for (M1=0; M1<=2*L; M1++){
      for (M2=0; M2<=2*L; M2++){

        Ctmpx = Complex(0.0,0.0);
        Ctmpy = Complex(0.0,0.0);
        Ctmpz = Complex(0.0,0.0);

        for (M3=0; M3<=2*L; M3++){

          Ctmp1 = Conjg(Comp2Real[L][M1][M3]);

          Ctmpx.r += CLx1[L][M3][M2].r*Ctmp1.r - CLx1[L][M3][M2].i*Ctmp1.i;
          Ctmpx.i += CLx1[L][M3][M2].r*Ctmp1.i + CLx1[L][M3][M2].i*Ctmp1.r;

          Ctmpy.r += CLy1[L][M3][M2].r*Ctmp1.r - CLy1[L][M3][M2].i*Ctmp1.i;
          Ctmpy.i += CLy1[L][M3][M2].r*Ctmp1.i + CLy1[L][M3][M2].i*Ctmp1.r;

          Ctmpz.r += CLz1[L][M3][M2].r*Ctmp1.r - CLz1[L][M3][M2].i*Ctmp1.i;
          Ctmpz.i += CLz1[L][M3][M2].r*Ctmp1.i + CLz1[L][M3][M2].i*Ctmp1.r;
	}

        Lx0[L][M1][M2] = Ctmpx.i;
        Ly0[L][M1][M2] = Ctmpy.i;
        Lz0[L][M1][M2] = Ctmpz.i;
      }
    }
  }

  /* freeing of arrays */

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLx[L][M]);
    }
    free(CLx[L]);
  }
  free(CLx);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLy[L][M]);
    }
    free(CLy[L]);
  }
  free(CLy);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLz[L][M]);
    }
    free(CLz[L]);
  }
  free(CLz);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLx1[L][M]);
    }
    free(CLx1[L]);
  }
  free(CLx1);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLy1[L][M]);
    }
    free(CLy1[L]);
  }
  free(CLy1);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLz1[L][M]);
    }
    free(CLz1[L]);
  }
  free(CLz1);

}
