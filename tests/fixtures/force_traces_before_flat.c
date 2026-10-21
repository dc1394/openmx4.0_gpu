/* Frozen HNL force traces from d437109f, also shared by core-hole.
   Retain the original element order, pair orientation and spin algebra. */
static const int reference_modes[13]={0,1,3,3,3,0,1,3,3,3,0,1,3};
static void reference_trace(int number)
{
  int i,j;
  switch(number) {
    case 0: {
      for (i = 0; i < ian; i++) {
          for (j = 0; j < jan; j++) {

              dEx += pref * CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r;
              dEy += pref * CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r;
              dEz += pref * CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r;
          }
      }
      break;
    }
    case 1: {
      for (i = 0; i < atomic->species.contracted_orbitals[Hwan]; i++) {
          for (j = 0; j < atomic->species.contracted_orbitals[Qwan]; j++) {

              dEx += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r);
              dEy += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r);
              dEz += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r);
          }
      }
      break;
    }
    case 2: {
      for (i = 0; i < atomic->species.contracted_orbitals[Hwan]; i++) {
          for (j = 0; j < atomic->species.contracted_orbitals[Qwan]; j++) {

              dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

              dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

              dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;
          }
      }
      break;
    }
    case 3: {
      for (i = 0; i < atomic->species.contracted_orbitals[Hwan]; i++) { /* Hwan */
          for (j = 0; j < atomic->species.contracted_orbitals[Qwan]; j++) { /* Qwan  */

              dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

              dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

              dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;

          } /* j */
      }
      break;
    }
    case 4: {
      for (i = 0; i < atomic->species.contracted_orbitals[Qwan]; i++) { /* Qwan */
          for (j = 0; j < atomic->species.contracted_orbitals[Hwan]; j++) { /* Hwan */

              dEx += CDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].r
                  - iDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].i
                  + CDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].r
                  - iDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].i
                  + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hx1[2][i][j].r
                  - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hx1[2][i][j].i;

              dEy += CDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].r
                  - iDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].i
                  + CDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].r
                  - iDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].i
                  + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hy1[2][i][j].r
                  - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hy1[2][i][j].i;

              dEz += CDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].r
                  - iDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].i
                  + CDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].r
                  - iDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].i
                  + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hz1[2][i][j].r
                  - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hz1[2][i][j].i;

          } /* j */
      }
      break;
    }
    case 5: {
      for (i = 0; i < ian; i++) {
          for (j = 0; j < jan; j++) {

              dEx += pref * CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r;
              dEy += pref * CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r;
              dEz += pref * CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r;
          }
      }
      break;
    }
    case 6: {
      for (i = 0; i < ian; i++) {
          for (j = 0; j < jan; j++) {

              dEx += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r);
              dEy += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r);
              dEz += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r);
          }
      }
      break;
    }
    case 7: {
      for (i = 0; i < atomic->species.contracted_orbitals[Hwan]; i++) {
          for (j = 0; j < atomic->species.contracted_orbitals[Qwan]; j++) {

              dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

              dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

              dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;
          }
      }
      break;
    }
    case 8: {
      for (i = 0; i < atomic->species.contracted_orbitals[Hwan]; i++) { /* Hwan */
          for (j = 0; j < atomic->species.contracted_orbitals[Qwan]; j++) { /* Qwan  */

              dEx += CDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hx0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hx0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx0[2][i][j].i;

              dEy += CDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hy0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hy0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy0[2][i][j].i;

              dEz += CDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].r
                  - iDM0[0][Mh_AN][kl][i][j] * Hz0[0][i][j].i
                  + CDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].r
                  - iDM0[1][Mh_AN][kl][i][j] * Hz0[1][i][j].i
                  + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz0[2][i][j].r
                  - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz0[2][i][j].i;

          } /* j */
      }
      break;
    }
    case 9: {
      for (i = 0; i < atomic->species.contracted_orbitals[Qwan]; i++) { /* Qwan */
          for (j = 0; j < atomic->species.contracted_orbitals[Hwan]; j++) { /* Hwan */

              dEx += CDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].r
                  - iDM0[0][Mq_AN][kl1][i][j] * Hx1[0][i][j].i
                  + CDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].r
                  - iDM0[1][Mq_AN][kl1][i][j] * Hx1[1][i][j].i
                  + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hx1[2][i][j].r
                  - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hx1[2][i][j].i;

              dEy += CDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].r
                  - iDM0[0][Mq_AN][kl1][i][j] * Hy1[0][i][j].i
                  + CDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].r
                  - iDM0[1][Mq_AN][kl1][i][j] * Hy1[1][i][j].i
                  + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hy1[2][i][j].r
                  - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hy1[2][i][j].i;

              dEz += CDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].r
                  - iDM0[0][Mq_AN][kl1][i][j] * Hz1[0][i][j].i
                  + CDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].r
                  - iDM0[1][Mq_AN][kl1][i][j] * Hz1[1][i][j].i
                  + 2.0 * CDM0[2][Mq_AN][kl1][i][j] * Hz1[2][i][j].r
                  - 2.0 * CDM0[3][Mq_AN][kl1][i][j] * Hz1[2][i][j].i;

          } /* j */
      }
      break;
    }
    case 10: {
      for (i = 0; i < ian; i++) {
          for (j = 0; j < jan; j++) {
              dEx_threads[OMPID] += pref * CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r;
              dEy_threads[OMPID] += pref * CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r;
              dEz_threads[OMPID] += pref * CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r;
          }
      }
      break;
    }
    case 11: {
      for (i = 0; i < ian; i++) {
          for (j = 0; j < jan; j++) {

              dEx_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].r);
              dEy_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].r);
              dEz_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r + CDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].r);
          }
      }
      break;
    }
    case 12: {
      for (i = 0; i < ian; i++) {
          for (j = 0; j < jan; j++) {

              dEx_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].r - iDM0[0][Mh_AN][kl][i][j] * Hx[0][i][j].i + CDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].r - iDM0[1][Mh_AN][kl][i][j] * Hx[1][i][j].i + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hx[2][i][j].r - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hx[2][i][j].i);

              dEy_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].r - iDM0[0][Mh_AN][kl][i][j] * Hy[0][i][j].i + CDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].r - iDM0[1][Mh_AN][kl][i][j] * Hy[1][i][j].i + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hy[2][i][j].r - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hy[2][i][j].i);

              dEz_threads[OMPID] += pref * (CDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].r - iDM0[0][Mh_AN][kl][i][j] * Hz[0][i][j].i + CDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].r - iDM0[1][Mh_AN][kl][i][j] * Hz[1][i][j].i + 2.0 * CDM0[2][Mh_AN][kl][i][j] * Hz[2][i][j].r - 2.0 * CDM0[3][Mh_AN][kl][i][j] * Hz[2][i][j].i);
          }
      }
      break;
    }
    default: abort();
  }
}
