/* Frozen Set_Comp2Real from aeadecf0, unchanged arithmetic and row order.
   Test-only caller supplies zeroed Comp2Real, YOUSO36 and Complex(). */
void Set_Comp2Real()
{
  int i,j,L;

  /* s */

  Comp2Real[0][0][0] = Complex( 1.0, 0.0);

  /* p */

  Comp2Real[1][0][0] = Complex( 1.0/sqrt(2.0), 0.0);
  Comp2Real[1][0][1] = Complex( 0.0,           0.0);
  Comp2Real[1][0][2] = Complex(-1.0/sqrt(2.0), 0.0);

  Comp2Real[1][1][0] = Complex( 0.0, 1.0/sqrt(2.0));
  Comp2Real[1][1][1] = Complex( 0.0,          0.0);
  Comp2Real[1][1][2] = Complex( 0.0, 1.0/sqrt(2.0));

  Comp2Real[1][2][0] = Complex( 0.0, 0.0);
  Comp2Real[1][2][1] = Complex( 1.0, 0.0);
  Comp2Real[1][2][2] = Complex( 0.0, 0.0);

  /* d */

  Comp2Real[2][0][0] = Complex( 0.0,           0.0);
  Comp2Real[2][0][1] = Complex( 0.0,           0.0);
  Comp2Real[2][0][2] = Complex( 1.0,           0.0);
  Comp2Real[2][0][3] = Complex( 0.0,           0.0);
  Comp2Real[2][0][4] = Complex( 0.0,           0.0);

  Comp2Real[2][1][0] = Complex( 1.0/sqrt(2.0), 0.0);
  Comp2Real[2][1][1] = Complex( 0.0,           0.0);
  Comp2Real[2][1][2] = Complex( 0.0,           0.0);
  Comp2Real[2][1][3] = Complex( 0.0,           0.0);
  Comp2Real[2][1][4] = Complex( 1.0/sqrt(2.0), 0.0);

  Comp2Real[2][2][0] = Complex( 0.0, 1.0/sqrt(2.0));
  Comp2Real[2][2][1] = Complex( 0.0,           0.0);
  Comp2Real[2][2][2] = Complex( 0.0,           0.0);
  Comp2Real[2][2][3] = Complex( 0.0,           0.0);
  Comp2Real[2][2][4] = Complex( 0.0,-1.0/sqrt(2.0));

  Comp2Real[2][3][0] = Complex( 0.0,           0.0);
  Comp2Real[2][3][1] = Complex( 1.0/sqrt(2.0), 0.0);
  Comp2Real[2][3][2] = Complex( 0.0,           0.0);
  Comp2Real[2][3][3] = Complex(-1.0/sqrt(2.0), 0.0);
  Comp2Real[2][3][4] = Complex( 0.0,           0.0);

  Comp2Real[2][4][0] = Complex( 0.0,           0.0);
  Comp2Real[2][4][1] = Complex( 0.0, 1.0/sqrt(2.0));
  Comp2Real[2][4][2] = Complex( 0.0,           0.0);
  Comp2Real[2][4][3] = Complex( 0.0, 1.0/sqrt(2.0));
  Comp2Real[2][4][4] = Complex( 0.0,           0.0);

  /* f */

  Comp2Real[3][0][0] = Complex( 0.0,           0.0);
  Comp2Real[3][0][1] = Complex( 0.0,           0.0);
  Comp2Real[3][0][2] = Complex( 0.0,           0.0);
  Comp2Real[3][0][3] = Complex( 1.0,           0.0);
  Comp2Real[3][0][4] = Complex( 0.0,           0.0);
  Comp2Real[3][0][5] = Complex( 0.0,           0.0);
  Comp2Real[3][0][6] = Complex( 0.0,           0.0);

  Comp2Real[3][1][0] = Complex( 0.0,           0.0);
  Comp2Real[3][1][1] = Complex( 0.0,           0.0);
  Comp2Real[3][1][2] = Complex( 1.0/sqrt(2.0), 0.0);
  Comp2Real[3][1][3] = Complex( 0.0,           0.0);
  Comp2Real[3][1][4] = Complex(-1.0/sqrt(2.0), 0.0);
  Comp2Real[3][1][5] = Complex( 0.0,           0.0);
  Comp2Real[3][1][6] = Complex( 0.0,           0.0);

  Comp2Real[3][2][0] = Complex( 0.0,           0.0);
  Comp2Real[3][2][1] = Complex( 0.0,           0.0);
  Comp2Real[3][2][2] = Complex( 0.0, 1.0/sqrt(2.0));
  Comp2Real[3][2][3] = Complex( 0.0,           0.0);
  Comp2Real[3][2][4] = Complex( 0.0, 1.0/sqrt(2.0));
  Comp2Real[3][2][5] = Complex( 0.0,           0.0);
  Comp2Real[3][2][6] = Complex( 0.0,           0.0);

  Comp2Real[3][3][0] = Complex( 0.0,           0.0);
  Comp2Real[3][3][1] = Complex( 1.0/sqrt(2.0), 0.0);
  Comp2Real[3][3][2] = Complex( 0.0,           0.0);
  Comp2Real[3][3][3] = Complex( 0.0,           0.0);
  Comp2Real[3][3][4] = Complex( 0.0,           0.0);
  Comp2Real[3][3][5] = Complex( 1.0/sqrt(2.0), 0.0);
  Comp2Real[3][3][6] = Complex( 0.0,           0.0);

  Comp2Real[3][4][0] = Complex( 0.0,           0.0);
  Comp2Real[3][4][1] = Complex( 0.0, 1.0/sqrt(2.0));
  Comp2Real[3][4][2] = Complex( 0.0,           0.0);
  Comp2Real[3][4][3] = Complex( 0.0,           0.0);
  Comp2Real[3][4][4] = Complex( 0.0,           0.0);
  Comp2Real[3][4][5] = Complex( 0.0,-1.0/sqrt(2.0));
  Comp2Real[3][4][6] = Complex( 0.0,           0.0);

  Comp2Real[3][5][0] = Complex( 1.0/sqrt(2.0), 0.0);
  Comp2Real[3][5][1] = Complex( 0.0,           0.0);
  Comp2Real[3][5][2] = Complex( 0.0,           0.0);
  Comp2Real[3][5][3] = Complex( 0.0,           0.0);
  Comp2Real[3][5][4] = Complex( 0.0,           0.0);
  Comp2Real[3][5][5] = Complex( 0.0,           0.0);
  Comp2Real[3][5][6] = Complex(-1.0/sqrt(2.0), 0.0);

  Comp2Real[3][6][0] = Complex( 0.0, 1.0/sqrt(2.0));
  Comp2Real[3][6][1] = Complex( 0.0,           0.0);
  Comp2Real[3][6][2] = Complex( 0.0,           0.0);
  Comp2Real[3][6][3] = Complex( 0.0,           0.0);
  Comp2Real[3][6][4] = Complex( 0.0,           0.0);
  Comp2Real[3][6][5] = Complex( 0.0,           0.0);
  Comp2Real[3][6][6] = Complex( 0.0, 1.0/sqrt(2.0));

  /* g(4), h(5), i(6), j(7), k(8), l(9), and m(10) */

  for (L=4; L<=YOUSO36; L++){

    for (i=0; i<(2*L+1); i++){
      for (j=0; j<(2*L+1); j++){
	Comp2Real[L][i][j]  = Complex(0.0, 0.0);
      }
    }

    Comp2Real[L][0][L] = Complex(1.0, 0.0);

    j = -1;
    for (i=1; i<(L*2+1); i=i+4){
      j++;
      Comp2Real[L][i][L-(2*j+1)]  = Complex( 1.0/sqrt(2.0), 0.0);
      Comp2Real[L][i][L+(2*j+1)]  = Complex(-1.0/sqrt(2.0), 0.0);
    }

    j = 0;
    for (i=3; i<(L*2+1); i=i+4){
      j++;
      Comp2Real[L][i][L-2*j]  = Complex(1.0/sqrt(2.0), 0.0);
      Comp2Real[L][i][L+2*j]  = Complex(1.0/sqrt(2.0), 0.0);
    }

    j = -1;
    for (i=2; i<(L*2+1); i=i+4){
      j++;
      Comp2Real[L][i][L-(2*j+1)]  = Complex(0.0, 1.0/sqrt(2.0));
      Comp2Real[L][i][L+(2*j+1)]  = Complex(0.0, 1.0/sqrt(2.0));
    }

    j = 0;
    for (i=4; i<(L*2+1); i=i+4){
      j++;
      Comp2Real[L][i][L-2*j]  = Complex(0.0,  1.0/sqrt(2.0));
      Comp2Real[L][i][L+2*j]  = Complex(0.0, -1.0/sqrt(2.0));
    }
  }

}
