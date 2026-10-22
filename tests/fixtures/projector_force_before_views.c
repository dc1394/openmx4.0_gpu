/* Frozen from 6b57a7fc: scalar/spinor arithmetic and all nine rectangular HNL call sites. */
static void reference_dHNL_SO_rows(int spin,
    double* sumx0r,
    double* sumy0r,
    double* sumz0r,
    double* sumx1r,
    double* sumy1r,
    double* sumz1r,
    double* sumx2r,
    double* sumy2r,
    double* sumz2r,
    double* sumx0i,
    double* sumy0i,
    double* sumz0i,
    double* sumx1i,
    double* sumy1i,
    double* sumz1i,
    double* sumx2i,
    double* sumy2i,
    double* sumz2i,
    double fugou,
    double PFp,
    double PFm,
    double ene_p,
    double ene_m,
    int l2, int* l,
    const double* restrict A0x, const double* restrict A0y, const double* restrict A0z,
    const double* restrict A1x, const double* restrict A1y, const double* restrict A1z,
    const double* restrict B0, const double* restrict B1)
{
    const double* restrict A0v[4] = { NULL, A0x, A0y, A0z };
    const double* restrict A1v[4] = { NULL, A1x, A1y, A1z };

    int l3, i;
    double tmpx, tmpy, tmpz;
    double tmp0, tmp1, tmp2;
    double tmp3, tmp4, tmp5, tmp6;
    double deri[4];

    /****************************************************
      off-diagonal contribution to up-dn matrix
      for spin non-collinear
    ****************************************************/

    if (spin == 3) {

        /* p */
        if (l2 == 2) {

            /* real contribution of l+1/2 to off diagonal up-down matrix */
            tmpx = fugou * (ene_p / 3.0 * A0x[*l] * B0[*l + 2] - ene_p / 3.0 * A0x[*l + 2] * B0[*l]);

            tmpy = fugou * (ene_p / 3.0 * A0y[*l] * B0[*l + 2] - ene_p / 3.0 * A0y[*l + 2] * B0[*l]);

            tmpz = fugou * (ene_p / 3.0 * A0z[*l] * B0[*l + 2] - ene_p / 3.0 * A0z[*l + 2] * B0[*l]);

            *sumx2r += tmpx;
            *sumy2r += tmpy;
            *sumz2r += tmpz;

            /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

            tmpx = fugou * (-ene_p / 3.0 * A0x[*l + 1] * B0[*l + 2] + ene_p / 3.0 * A0x[*l + 2] * B0[*l + 1]);

            tmpy = fugou * (-ene_p / 3.0 * A0y[*l + 1] * B0[*l + 2] + ene_p / 3.0 * A0y[*l + 2] * B0[*l + 1]);

            tmpz = fugou * (-ene_p / 3.0 * A0z[*l + 1] * B0[*l + 2] + ene_p / 3.0 * A0z[*l + 2] * B0[*l + 1]);

            *sumx2i += tmpx;
            *sumy2i += tmpy;
            *sumz2i += tmpz;

            /* real contribution of l-1/2 for to diagonal up-down matrix */

            tmpx = fugou * (ene_m / 3.0 * A1x[*l] * B1[*l + 2] - ene_m / 3.0 * A1x[*l + 2] * B1[*l]);

            tmpy = fugou * (ene_m / 3.0 * A1y[*l] * B1[*l + 2] - ene_m / 3.0 * A1y[*l + 2] * B1[*l]);

            tmpz = fugou * (ene_m / 3.0 * A1z[*l] * B1[*l + 2] - ene_m / 3.0 * A1z[*l + 2] * B1[*l]);

            *sumx2r -= tmpx;
            *sumy2r -= tmpy;
            *sumz2r -= tmpz;

            /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

            tmpx = fugou * (-ene_m / 3.0 * A1x[*l + 1] * B1[*l + 2] + ene_m / 3.0 * A1x[*l + 2] * B1[*l + 1]);

            tmpy = fugou * (-ene_m / 3.0 * A1y[*l + 1] * B1[*l + 2] + ene_m / 3.0 * A1y[*l + 2] * B1[*l + 1]);

            tmpz = fugou * (-ene_m / 3.0 * A1z[*l + 1] * B1[*l + 2] + ene_m / 3.0 * A1z[*l + 2] * B1[*l + 1]);

            *sumx2i -= tmpx;
            *sumy2i -= tmpy;
            *sumz2i -= tmpz;
        }

        /* d */
        if (l2 == 4) {

            tmp0 = sqrt(3.0);
            tmp1 = ene_p / 5.0;
            tmp2 = tmp0 * tmp1;

            /* real contribution of l+1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp2 * A0v[i][*l] * B0[*l + 3] + tmp2 * A0v[i][*l + 3] * B0[*l] + tmp1 * A0v[i][*l + 1] * B0[*l + 3] - tmp1 * A0v[i][*l + 3] * B0[*l + 1] + tmp1 * A0v[i][*l + 2] * B0[*l + 4] - tmp1 * A0v[i][*l + 4] * B0[*l + 2]);
            }
            *sumx2r += deri[1];
            *sumy2r += deri[2];
            *sumz2r += deri[3];

            /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp2 * A0v[i][*l] * B0[*l + 4] - tmp2 * A0v[i][*l + 4] * B0[*l] + tmp1 * A0v[i][*l + 1] * B0[*l + 4] - tmp1 * A0v[i][*l + 4] * B0[*l + 1] - tmp1 * A0v[i][*l + 2] * B0[*l + 3] + tmp1 * A0v[i][*l + 3] * B0[*l + 2]);
            }
            *sumx2i += deri[1];
            *sumy2i += deri[2];
            *sumz2i += deri[3];

            /* real contribution of l-1/2 for to diagonal up-down matrix */

            tmp1 = ene_m / 5.0;
            tmp2 = tmp0 * tmp1;

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp2 * A1v[i][*l] * B1[*l + 3] + tmp2 * A1v[i][*l + 3] * B1[*l] + tmp1 * A1v[i][*l + 1] * B1[*l + 3] - tmp1 * A1v[i][*l + 3] * B1[*l + 1] + tmp1 * A1v[i][*l + 2] * B1[*l + 4] - tmp1 * A1v[i][*l + 4] * B1[*l + 2]);
            }
            *sumx2r -= deri[1];
            *sumy2r -= deri[2];
            *sumz2r -= deri[3];

            /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp2 * A1v[i][*l] * B1[*l + 4] - tmp2 * A1v[i][*l + 4] * B1[*l] + tmp1 * A1v[i][*l + 1] * B1[*l + 4] - tmp1 * A1v[i][*l + 4] * B1[*l + 1] - tmp1 * A1v[i][*l + 2] * B1[*l + 3] + tmp1 * A1v[i][*l + 3] * B1[*l + 2]);
            }
            *sumx2i -= deri[1];
            *sumy2i -= deri[2];
            *sumz2i -= deri[3];
        }

        /* f */
        if (l2 == 6) {

            /* real contribution of l+1/2 to off diagonal up-down matrix */

            tmp0 = sqrt(6.0);
            tmp1 = sqrt(3.0 / 2.0);
            tmp2 = sqrt(5.0 / 2.0);

            tmp3 = ene_p / 7.0;
            tmp4 = tmp1 * tmp3; /* sqrt(3.0/2.0) */
            tmp5 = tmp2 * tmp3; /* sqrt(5.0/2.0) */
            tmp6 = tmp0 * tmp3; /* sqrt(6.0)     */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp6 * A0v[i][*l] * B0[*l + 1] + tmp6 * A0v[i][*l + 1] * B0[*l] - tmp5 * A0v[i][*l + 1] * B0[*l + 3] + tmp5 * A0v[i][*l + 3] * B0[*l + 1] - tmp5 * A0v[i][*l + 2] * B0[*l + 4] + tmp5 * A0v[i][*l + 4] * B0[*l + 2] - tmp4 * A0v[i][*l + 3] * B0[*l + 5] + tmp4 * A0v[i][*l + 5] * B0[*l + 3] - tmp4 * A0v[i][*l + 4] * B0[*l + 6] + tmp4 * A0v[i][*l + 6] * B0[*l + 4]);
            }
            *sumx2r += deri[1];
            *sumy2r += deri[2];
            *sumz2r += deri[3];

            /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp6 * A0v[i][*l] * B0[*l + 2] - tmp6 * A0v[i][*l + 2] * B0[*l] + tmp5 * A0v[i][*l + 1] * B0[*l + 4] - tmp5 * A0v[i][*l + 4] * B0[*l + 1] - tmp5 * A0v[i][*l + 2] * B0[*l + 3] + tmp5 * A0v[i][*l + 3] * B0[*l + 2] + tmp4 * A0v[i][*l + 3] * B0[*l + 6] - tmp4 * A0v[i][*l + 6] * B0[*l + 3] - tmp4 * A0v[i][*l + 4] * B0[*l + 5] + tmp4 * A0v[i][*l + 5] * B0[*l + 4]);
            }
            *sumx2i += deri[1];
            *sumy2i += deri[2];
            *sumz2i += deri[3];

            /* real contribution of l-1/2 for to diagonal up-down matrix */

            tmp3 = ene_m / 7.0;
            tmp4 = tmp1 * tmp3; /* sqrt(3.0/2.0) */
            tmp5 = tmp2 * tmp3; /* sqrt(5.0/2.0) */
            tmp6 = tmp0 * tmp3; /* sqrt(6.0)     */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp6 * A1v[i][*l] * B1[*l + 1] + tmp6 * A1v[i][*l + 1] * B1[*l] - tmp5 * A1v[i][*l + 1] * B1[*l + 3] + tmp5 * A1v[i][*l + 3] * B1[*l + 1] - tmp5 * A1v[i][*l + 2] * B1[*l + 4] + tmp5 * A1v[i][*l + 4] * B1[*l + 2] - tmp4 * A1v[i][*l + 3] * B1[*l + 5] + tmp4 * A1v[i][*l + 5] * B1[*l + 3] - tmp4 * A1v[i][*l + 4] * B1[*l + 6] + tmp4 * A1v[i][*l + 6] * B1[*l + 4]);
            }
            *sumx2r -= deri[1];
            *sumy2r -= deri[2];
            *sumz2r -= deri[3];

            /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp6 * A1v[i][*l] * B1[*l + 2] - tmp6 * A1v[i][*l + 2] * B1[*l] + tmp5 * A1v[i][*l + 1] * B1[*l + 4] - tmp5 * A1v[i][*l + 4] * B1[*l + 1] - tmp5 * A1v[i][*l + 2] * B1[*l + 3] + tmp5 * A1v[i][*l + 3] * B1[*l + 2] + tmp4 * A1v[i][*l + 3] * B1[*l + 6] - tmp4 * A1v[i][*l + 6] * B1[*l + 3] - tmp4 * A1v[i][*l + 4] * B1[*l + 5] + tmp4 * A1v[i][*l + 5] * B1[*l + 4]);
            }
            *sumx2i -= deri[1];
            *sumy2i -= deri[2];
            *sumz2i -= deri[3];
        }
    }

    /****************************************************
        off-diagonal contribution on up-up and dn-dn
    ****************************************************/

    /* p */
    if (l2 == 2) {

        tmpx = fugou * (ene_p / 3.0 * A0x[*l] * B0[*l + 1] - ene_p / 3.0 * A0x[*l + 1] * B0[*l]);

        tmpy = fugou * (ene_p / 3.0 * A0y[*l] * B0[*l + 1] - ene_p / 3.0 * A0y[*l + 1] * B0[*l]);

        tmpz = fugou * (ene_p / 3.0 * A0z[*l] * B0[*l + 1] - ene_p / 3.0 * A0z[*l + 1] * B0[*l]);

        /* contribution of l+1/2 for up spin */
        *sumx0i += -tmpx;
        *sumy0i += -tmpy;
        *sumz0i += -tmpz;

        /* contribution of l+1/2 for down spin */
        *sumx1i += tmpx;
        *sumy1i += tmpy;
        *sumz1i += tmpz;

        tmpx = fugou * (ene_m / 3.0 * A1x[*l] * B1[*l + 1] - ene_m / 3.0 * A1x[*l + 1] * B1[*l]);

        tmpy = fugou * (ene_m / 3.0 * A1y[*l] * B1[*l + 1] - ene_m / 3.0 * A1y[*l + 1] * B1[*l]);

        tmpz = fugou * (ene_m / 3.0 * A1z[*l] * B1[*l + 1] - ene_m / 3.0 * A1z[*l + 1] * B1[*l]);

        /* contribution of l-1/2 for up spin */
        *sumx0i += tmpx;
        *sumy0i += tmpy;
        *sumz0i += tmpz;

        /* contribution of l+1/2 for down spin */
        *sumx1i += -tmpx;
        *sumy1i += -tmpy;
        *sumz1i += -tmpz;
    }

    /* d */
    else if (l2 == 4) {

        tmpx = fugou * (ene_p * 2.0 / 5.0 * A0x[*l + 1] * B0[*l + 2] - ene_p * 2.0 / 5.0 * A0x[*l + 2] * B0[*l + 1] + ene_p * 1.0 / 5.0 * A0x[*l + 3] * B0[*l + 4] - ene_p * 1.0 / 5.0 * A0x[*l + 4] * B0[*l + 3]);

        tmpy = fugou * (ene_p * 2.0 / 5.0 * A0y[*l + 1] * B0[*l + 2] - ene_p * 2.0 / 5.0 * A0y[*l + 2] * B0[*l + 1] + ene_p * 1.0 / 5.0 * A0y[*l + 3] * B0[*l + 4] - ene_p * 1.0 / 5.0 * A0y[*l + 4] * B0[*l + 3]);

        tmpz = fugou * (ene_p * 2.0 / 5.0 * A0z[*l + 1] * B0[*l + 2] - ene_p * 2.0 / 5.0 * A0z[*l + 2] * B0[*l + 1] + ene_p * 1.0 / 5.0 * A0z[*l + 3] * B0[*l + 4] - ene_p * 1.0 / 5.0 * A0z[*l + 4] * B0[*l + 3]);

        /* contribution of l+1/2 for up spin */
        *sumx0i += -tmpx;
        *sumy0i += -tmpy;
        *sumz0i += -tmpz;

        /* contribution of l+1/2 for down spin */
        *sumx1i += tmpx;
        *sumy1i += tmpy;
        *sumz1i += tmpz;

        tmpx = fugou * (ene_m * 2.0 / 5.0 * A1x[*l + 1] * B1[*l + 2] - ene_m * 2.0 / 5.0 * A1x[*l + 2] * B1[*l + 1] + ene_m * 1.0 / 5.0 * A1x[*l + 3] * B1[*l + 4] - ene_m * 1.0 / 5.0 * A1x[*l + 4] * B1[*l + 3]);

        tmpy = fugou * (ene_m * 2.0 / 5.0 * A1y[*l + 1] * B1[*l + 2] - ene_m * 2.0 / 5.0 * A1y[*l + 2] * B1[*l + 1] + ene_m * 1.0 / 5.0 * A1y[*l + 3] * B1[*l + 4] - ene_m * 1.0 / 5.0 * A1y[*l + 4] * B1[*l + 3]);

        tmpz = fugou * (ene_m * 2.0 / 5.0 * A1z[*l + 1] * B1[*l + 2] - ene_m * 2.0 / 5.0 * A1z[*l + 2] * B1[*l + 1] + ene_m * 1.0 / 5.0 * A1z[*l + 3] * B1[*l + 4] - ene_m * 1.0 / 5.0 * A1z[*l + 4] * B1[*l + 3]);

        /* contribution of l-1/2 for up spin */
        *sumx0i += tmpx;
        *sumy0i += tmpy;
        *sumz0i += tmpz;

        /* contribution of l-1/2 for down spin */
        *sumx1i += -tmpx;
        *sumy1i += -tmpy;
        *sumz1i += -tmpz;

    }

    /* f */
    else if (l2 == 6) {

        tmpx = fugou * (ene_p * 1.0 / 7.0 * A0x[*l + 1] * B0[*l + 2] - ene_p * 1.0 / 7.0 * A0x[*l + 2] * B0[*l + 1] + ene_p * 2.0 / 7.0 * A0x[*l + 3] * B0[*l + 4] - ene_p * 2.0 / 7.0 * A0x[*l + 4] * B0[*l + 3] + ene_p * 3.0 / 7.0 * A0x[*l + 5] * B0[*l + 6] - ene_p * 3.0 / 7.0 * A0x[*l + 6] * B0[*l + 5]);

        tmpy = fugou * (ene_p * 1.0 / 7.0 * A0y[*l + 1] * B0[*l + 2] - ene_p * 1.0 / 7.0 * A0y[*l + 2] * B0[*l + 1] + ene_p * 2.0 / 7.0 * A0y[*l + 3] * B0[*l + 4] - ene_p * 2.0 / 7.0 * A0y[*l + 4] * B0[*l + 3] + ene_p * 3.0 / 7.0 * A0y[*l + 5] * B0[*l + 6] - ene_p * 3.0 / 7.0 * A0y[*l + 6] * B0[*l + 5]);

        tmpz = fugou * (ene_p * 1.0 / 7.0 * A0z[*l + 1] * B0[*l + 2] - ene_p * 1.0 / 7.0 * A0z[*l + 2] * B0[*l + 1] + ene_p * 2.0 / 7.0 * A0z[*l + 3] * B0[*l + 4] - ene_p * 2.0 / 7.0 * A0z[*l + 4] * B0[*l + 3] + ene_p * 3.0 / 7.0 * A0z[*l + 5] * B0[*l + 6] - ene_p * 3.0 / 7.0 * A0z[*l + 6] * B0[*l + 5]);

        /* contribution of l+1/2 for up spin */
        *sumx0i += -tmpx;
        *sumy0i += -tmpy;
        *sumz0i += -tmpz;

        /* contribution of l+1/2 for down spin */
        *sumx1i += tmpx;
        *sumy1i += tmpy;
        *sumz1i += tmpz;

        tmpx = fugou * (ene_m * 1.0 / 7.0 * A1x[*l + 1] * B1[*l + 2] - ene_m * 1.0 / 7.0 * A1x[*l + 2] * B1[*l + 1] + ene_m * 2.0 / 7.0 * A1x[*l + 3] * B1[*l + 4] - ene_m * 2.0 / 7.0 * A1x[*l + 4] * B1[*l + 3] + ene_m * 3.0 / 7.0 * A1x[*l + 5] * B1[*l + 6] - ene_m * 3.0 / 7.0 * A1x[*l + 6] * B1[*l + 5]);

        tmpy = fugou * (ene_m * 1.0 / 7.0 * A1y[*l + 1] * B1[*l + 2] - ene_m * 1.0 / 7.0 * A1y[*l + 2] * B1[*l + 1] + ene_m * 2.0 / 7.0 * A1y[*l + 3] * B1[*l + 4] - ene_m * 2.0 / 7.0 * A1y[*l + 4] * B1[*l + 3] + ene_m * 3.0 / 7.0 * A1y[*l + 5] * B1[*l + 6] - ene_m * 3.0 / 7.0 * A1y[*l + 6] * B1[*l + 5]);

        tmpz = fugou * (ene_m * 1.0 / 7.0 * A1z[*l + 1] * B1[*l + 2] - ene_m * 1.0 / 7.0 * A1z[*l + 2] * B1[*l + 1] + ene_m * 2.0 / 7.0 * A1z[*l + 3] * B1[*l + 4] - ene_m * 2.0 / 7.0 * A1z[*l + 4] * B1[*l + 3] + ene_m * 3.0 / 7.0 * A1z[*l + 5] * B1[*l + 6] - ene_m * 3.0 / 7.0 * A1z[*l + 6] * B1[*l + 5]);

        /* contribution of l-1/2 for up spin */
        *sumx0i += tmpx;
        *sumy0i += tmpy;
        *sumz0i += tmpz;

        /* contribution of l-1/2 for down spin */
        *sumx1i += -tmpx;
        *sumy1i += -tmpy;
        *sumz1i += -tmpz;
    }

    /****************************************************
        diagonal contribution on up-up and dn-dn
    ****************************************************/

    for (l3 = 0; l3 <= l2; l3++) {

        /* VNL for j=l+1/2 */

        tmpx = PFp * ene_p * A0x[*l] * B0[*l];
        tmpy = PFp * ene_p * A0y[*l] * B0[*l];
        tmpz = PFp * ene_p * A0z[*l] * B0[*l];

        *sumx0r += tmpx;
        *sumy0r += tmpy;
        *sumz0r += tmpz;

        *sumx1r += tmpx;
        *sumy1r += tmpy;
        *sumz1r += tmpz;

        /* VNL for j=l-1/2 */

        tmpx = PFm * ene_m * A1x[*l] * B1[*l];
        tmpy = PFm * ene_m * A1y[*l] * B1[*l];
        tmpz = PFm * ene_m * A1z[*l] * B1[*l];

        *sumx0r += tmpx;
        *sumy0r += tmpy;
        *sumz0r += tmpz;

        *sumx1r += tmpx;
        *sumy1r += tmpy;
        *sumz1r += tmpz;

        *l = *l + 1;
    }
}
static void legacy_block_call(int number) {
 int l,l1,l2,l3;double ene,ene_p,ene_m,PFp,PFm,sumx,sumy,sumz;
 dcomplex sumx0,sumy0,sumz0,sumx1,sumy1,sumz1,sumx2,sumy2,sumz2;
 switch(number) {
case 0: {
for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        l = 0;
                        for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                            ene = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                            if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0)
                                l2 = 0;
                            else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1)
                                l2 = 2;
                            else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2)
                                l2 = 4;
                            else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3)
                                l2 = 6;

                            if (Mj_AN <= Matomnum)
                                Mj_AN2 = Mj_AN;
                            else
                                Mj_AN2 = Matomnum + 1;

                            for (l3 = 0; l3 <= l2; l3++) {
                                sumx += ene * DS_NL1[0][1][Mc_AN][k][m][l] * DS_NL1[0][0][Mj_AN2][kl][n][l];
                                sumy += ene * DS_NL1[0][2][Mc_AN][k][m][l] * DS_NL1[0][0][Mj_AN2][kl][n][l];
                                sumz += ene * DS_NL1[0][3][Mc_AN][k][m][l] * DS_NL1[0][0][Mj_AN2][kl][n][l];
                                l++;
                            }
                        }

                        Hx[0][m][n].r += sumx;
                        Hy[0][m][n].r += sumy;
                        Hz[0][m][n].r += sumz;

                        Hx[1][m][n].r += sumx;
                        Hy[1][m][n].r += sumy;
                        Hz[1][m][n].r += sumz;

                    } /* n */
                }
break;}
case 1: {
for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);

                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);

                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        if (Mj_AN <= Matomnum)
                            Mj_AN2 = Mj_AN;
                        else
                            Mj_AN2 = Matomnum + 1;

                        l = 0;
                        for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                            ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                            ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1 - 1];

                            if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0) {
                                l2 = 0;
                                PFp = 1.0;
                                PFm = 0.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1) {
                                l2 = 2;
                                PFp = 2.0 / 3.0;
                                PFm = 1.0 / 3.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2) {
                                l2 = 4;
                                PFp = 3.0 / 5.0;
                                PFm = 2.0 / 5.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3) {
                                l2 = 6;
                                PFp = 4.0 / 7.0;
                                PFm = 3.0 / 7.0;
                            }

                            reference_dHNL_SO_rows(SpinP_switch,&sumx0.r, &sumy0.r, &sumz0.r,
                                &sumx1.r, &sumy1.r, &sumz1.r,
                                &sumx2.r, &sumy2.r, &sumz2.r,
                                &sumx0.i, &sumy0.i, &sumz0.i,
                                &sumx1.i, &sumy1.i, &sumz1.i,
                                &sumx2.i, &sumy2.i, &sumz2.i,
                                1.0,
                                PFp, PFm,
                                ene_p, ene_m,
                                l2, &l,
                                DS_NL1[0][1][Mc_AN][k][m], DS_NL1[0][2][Mc_AN][k][m], DS_NL1[0][3][Mc_AN][k][m],
                                DS_NL1[1][1][Mc_AN][k][m], DS_NL1[1][2][Mc_AN][k][m], DS_NL1[1][3][Mc_AN][k][m],
                                DS_NL1[0][0][Mj_AN2][kl][n], DS_NL1[1][0][Mj_AN2][kl][n]);
                        }

                        if (q_AN == 0) {

                            l = 0;
                            for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                                ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                                ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1 - 1];

                                if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0) {
                                    l2 = 0;
                                    PFp = 1.0;
                                    PFm = 0.0;
                                } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1) {
                                    l2 = 2;
                                    PFp = 2.0 / 3.0;
                                    PFm = 1.0 / 3.0;
                                } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2) {
                                    l2 = 4;
                                    PFp = 3.0 / 5.0;
                                    PFm = 2.0 / 5.0;
                                } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3) {
                                    l2 = 6;
                                    PFp = 4.0 / 7.0;
                                    PFm = 3.0 / 7.0;
                                }

                                reference_dHNL_SO_rows(SpinP_switch,&sumx0.r, &sumy0.r, &sumz0.r,
                                    &sumx1.r, &sumy1.r, &sumz1.r,
                                    &sumx2.r, &sumy2.r, &sumz2.r,
                                    &sumx0.i, &sumy0.i, &sumz0.i,
                                    &sumx1.i, &sumy1.i, &sumz1.i,
                                    &sumx2.i, &sumy2.i, &sumz2.i,
                                    -1.0,
                                    PFp, PFm,
                                    ene_p, ene_m,
                                    l2, &l,
                                    DS_NL1[0][1][Mj_AN2][kl][n], DS_NL1[0][2][Mj_AN2][kl][n], DS_NL1[0][3][Mj_AN2][kl][n],
                                    DS_NL1[1][1][Mj_AN2][kl][n], DS_NL1[1][2][Mj_AN2][kl][n], DS_NL1[1][3][Mj_AN2][kl][n],
                                    DS_NL1[0][0][Mc_AN][k][m], DS_NL1[1][0][Mc_AN][k][m]);
                            }
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
break;}
case 2: {
for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        if (Mj_AN <= Matomnum) {
                            Mj_AN2 = Mj_AN;
                            kl2 = RMI1[Mc_AN][q_AN][0];
                        } else {
                            Mj_AN2 = Matomnum + 1;
                            kl2 = RMI1[Mc_AN][0][q_AN];
                        }

                        l = 0;
                        for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                            ene = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                            if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0)
                                l2 = 0;
                            else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1)
                                l2 = 2;
                            else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2)
                                l2 = 4;
                            else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3)
                                l2 = 6;

                            for (l3 = 0; l3 <= l2; l3++) {

                                sumx -= ene * DS_NL1[0][0][Mc_AN][0][m][l] * DS_NL1[0][1][Mj_AN2][kl2][n][l];
                                sumy -= ene * DS_NL1[0][0][Mc_AN][0][m][l] * DS_NL1[0][2][Mj_AN2][kl2][n][l];
                                sumz -= ene * DS_NL1[0][0][Mc_AN][0][m][l] * DS_NL1[0][3][Mj_AN2][kl2][n][l];
                                l++;
                            }
                        }

                        Hx[0][m][n].r += sumx;
                        Hy[0][m][n].r += sumy;
                        Hz[0][m][n].r += sumz;

                        Hx[1][m][n].r += sumx;
                        Hy[1][m][n].r += sumy;
                        Hz[1][m][n].r += sumz;
                    }
                }
break;}
case 3: {
for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);

                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);

                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        if (Mj_AN <= Matomnum) {
                            Mj_AN2 = Mj_AN;
                            kl2 = RMI1[Mc_AN][q_AN][0];
                        } else {
                            Mj_AN2 = Matomnum + 1;
                            kl2 = RMI1[Mc_AN][0][q_AN];
                        }

                        l = 0;
                        for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                            ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                            ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1 - 1];

                            if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0) {
                                l2 = 0;
                                PFp = 1.0;
                                PFm = 0.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1) {
                                l2 = 2;
                                PFp = 2.0 / 3.0;
                                PFm = 1.0 / 3.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2) {
                                l2 = 4;
                                PFp = 3.0 / 5.0;
                                PFm = 2.0 / 5.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3) {
                                l2 = 6;
                                PFp = 4.0 / 7.0;
                                PFm = 3.0 / 7.0;
                            }

                            /* 1 */

                            reference_dHNL_SO_rows(SpinP_switch,&sumx0.r, &sumy0.r, &sumz0.r,
                                &sumx1.r, &sumy1.r, &sumz1.r,
                                &sumx2.r, &sumy2.r, &sumz2.r,
                                &sumx0.i, &sumy0.i, &sumz0.i,
                                &sumx1.i, &sumy1.i, &sumz1.i,
                                &sumx2.i, &sumy2.i, &sumz2.i,
                                -1.0,
                                PFp, PFm,
                                -ene_p, -ene_m,
                                l2, &l,
                                DS_NL1[0][1][Mj_AN2][kl2][n], DS_NL1[0][2][Mj_AN2][kl2][n], DS_NL1[0][3][Mj_AN2][kl2][n],
                                DS_NL1[1][1][Mj_AN2][kl2][n], DS_NL1[1][2][Mj_AN2][kl2][n], DS_NL1[1][3][Mj_AN2][kl2][n],
                                DS_NL1[0][0][Mc_AN][0][m], DS_NL1[1][0][Mc_AN][0][m]);
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
break;}
case 4: {
for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);

                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);

                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        l = 0;
                        for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                            ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                            ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1 - 1];

                            if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0) {
                                l2 = 0;
                                PFp = 1.0;
                                PFm = 0.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1) {
                                l2 = 2;
                                PFp = 2.0 / 3.0;
                                PFm = 1.0 / 3.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2) {
                                l2 = 4;
                                PFp = 3.0 / 5.0;
                                PFm = 2.0 / 5.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3) {
                                l2 = 6;
                                PFp = 4.0 / 7.0;
                                PFm = 3.0 / 7.0;
                            }

                            reference_dHNL_SO_rows(SpinP_switch,&sumx0.r, &sumy0.r, &sumz0.r,
                                &sumx1.r, &sumy1.r, &sumz1.r,
                                &sumx2.r, &sumy2.r, &sumz2.r,
                                &sumx0.i, &sumy0.i, &sumz0.i,
                                &sumx1.i, &sumy1.i, &sumz1.i,
                                &sumx2.i, &sumy2.i, &sumz2.i,
                                -1.0,
                                PFp, PFm,
                                ene_p, ene_m,
                                l2, &l,
                                DS_NL1[0][1][Mj_AN][k][n], DS_NL1[0][2][Mj_AN][k][n], DS_NL1[0][3][Mj_AN][k][n],
                                DS_NL1[1][1][Mj_AN][k][n], DS_NL1[1][2][Mj_AN][k][n], DS_NL1[1][3][Mj_AN][k][n],
                                DS_NL1[0][0][Mi_AN2][kl][m], DS_NL1[1][0][Mi_AN2][kl][m]);
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
break;}
case 5: {
for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    sumx = 0.0;
                    sumy = 0.0;
                    sumz = 0.0;

                    l = 0;
                    for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                        ene = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                        if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0)
                            l2 = 0;
                        else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1)
                            l2 = 2;
                        else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2)
                            l2 = 4;
                        else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3)
                            l2 = 6;

                        for (l3 = 0; l3 <= l2; l3++) {
                            sumx -= ene * DS_NL1[0][1][Matomnum + 1][kl1][m][l] * DS_NL1[0][0][Matomnum + 1][kl2][n][l];
                            sumy -= ene * DS_NL1[0][2][Matomnum + 1][kl1][m][l] * DS_NL1[0][0][Matomnum + 1][kl2][n][l];
                            sumz -= ene * DS_NL1[0][3][Matomnum + 1][kl1][m][l] * DS_NL1[0][0][Matomnum + 1][kl2][n][l];
                            l++;
                        }
                    }

                    Hx[0][m][n].r = sumx;
                    Hy[0][m][n].r = sumy;
                    Hz[0][m][n].r = sumz;

                    Hx[1][m][n].r = sumx;
                    Hy[1][m][n].r = sumy;
                    Hz[1][m][n].r = sumz;

                    Hx[2][m][n].r = 0.0;
                    Hy[2][m][n].r = 0.0;
                    Hz[2][m][n].r = 0.0;

                    Hx[0][m][n].i = 0.0;
                    Hy[0][m][n].i = 0.0;
                    Hz[0][m][n].i = 0.0;

                    Hx[1][m][n].i = 0.0;
                    Hy[1][m][n].i = 0.0;
                    Hz[1][m][n].i = 0.0;

                    Hx[2][m][n].i = 0.0;
                    Hy[2][m][n].i = 0.0;
                    Hz[2][m][n].i = 0.0;
                }
            }
break;}
case 6: {
for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    sumx0 = Complex(0.0, 0.0);
                    sumy0 = Complex(0.0, 0.0);
                    sumz0 = Complex(0.0, 0.0);

                    sumx1 = Complex(0.0, 0.0);
                    sumy1 = Complex(0.0, 0.0);
                    sumz1 = Complex(0.0, 0.0);

                    sumx2 = Complex(0.0, 0.0);
                    sumy2 = Complex(0.0, 0.0);
                    sumz2 = Complex(0.0, 0.0);

                    l = 0;
                    for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                        ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                        ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1 - 1];

                        if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0) {
                            l2 = 0;
                            PFp = 1.0;
                            PFm = 0.0;
                        } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1) {
                            l2 = 2;
                            PFp = 2.0 / 3.0;
                            PFm = 1.0 / 3.0;
                        } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2) {
                            l2 = 4;
                            PFp = 3.0 / 5.0;
                            PFm = 2.0 / 5.0;
                        } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3) {
                            l2 = 6;
                            PFp = 4.0 / 7.0;
                            PFm = 3.0 / 7.0;
                        }

                        /* 2 */

                        reference_dHNL_SO_rows(SpinP_switch,&sumx0.r, &sumy0.r, &sumz0.r,
                            &sumx1.r, &sumy1.r, &sumz1.r,
                            &sumx2.r, &sumy2.r, &sumz2.r,
                            &sumx0.i, &sumy0.i, &sumz0.i,
                            &sumx1.i, &sumy1.i, &sumz1.i,
                            &sumx2.i, &sumy2.i, &sumz2.i,
                            1.0,
                            PFp, PFm,
                            -ene_p, -ene_m,
                            l2, &l,
                            DS_NL1[0][1][Matomnum + 1][kl1][m], DS_NL1[0][2][Matomnum + 1][kl1][m], DS_NL1[0][3][Matomnum + 1][kl1][m],
                            DS_NL1[1][1][Matomnum + 1][kl1][m], DS_NL1[1][2][Matomnum + 1][kl1][m], DS_NL1[1][3][Matomnum + 1][kl1][m],
                            DS_NL1[0][0][Matomnum + 1][kl2][n], DS_NL1[1][0][Matomnum + 1][kl2][n]);
                    }

                    Hx[0][m][n].r = sumx0.r; /* up-up */
                    Hy[0][m][n].r = sumy0.r; /* up-up */
                    Hz[0][m][n].r = sumz0.r; /* up-up */

                    Hx[1][m][n].r = sumx1.r; /* dn-dn */
                    Hy[1][m][n].r = sumy1.r; /* dn-dn */
                    Hz[1][m][n].r = sumz1.r; /* dn-dn */

                    Hx[2][m][n].r = sumx2.r; /* up-dn */
                    Hy[2][m][n].r = sumy2.r; /* up-dn */
                    Hz[2][m][n].r = sumz2.r; /* up-dn */

                    Hx[0][m][n].i = sumx0.i; /* up-up */
                    Hy[0][m][n].i = sumy0.i; /* up-up */
                    Hz[0][m][n].i = sumz0.i; /* up-up */

                    Hx[1][m][n].i = sumx1.i; /* dn-dn */
                    Hy[1][m][n].i = sumy1.i; /* dn-dn */
                    Hz[1][m][n].i = sumz1.i; /* dn-dn */

                    Hx[2][m][n].i = sumx2.i; /* up-dn */
                    Hy[2][m][n].i = sumy2.i; /* up-dn */
                    Hz[2][m][n].i = sumz2.i; /* up-dn */
                }
            }
break;}
case 7: {
for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        l = 0;
                        for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                            ene = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                            if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0)
                                l2 = 0;
                            else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1)
                                l2 = 2;
                            else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2)
                                l2 = 4;
                            else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3)
                                l2 = 6;

                            for (l3 = 0; l3 <= l2; l3++) {
                                sumx -= ene * DS_NL1[0][0][Matomnum + 1][kl1][m][l] * DS_NL1[0][1][Matomnum + 1][kl2][n][l];
                                sumy -= ene * DS_NL1[0][0][Matomnum + 1][kl1][m][l] * DS_NL1[0][2][Matomnum + 1][kl2][n][l];
                                sumz -= ene * DS_NL1[0][0][Matomnum + 1][kl1][m][l] * DS_NL1[0][3][Matomnum + 1][kl2][n][l];
                                l++;
                            }
                        }

                        Hx[0][m][n].r += sumx;
                        Hy[0][m][n].r += sumy;
                        Hz[0][m][n].r += sumz;

                        Hx[1][m][n].r += sumx;
                        Hy[1][m][n].r += sumy;
                        Hz[1][m][n].r += sumz;
                    }
                }
break;}
case 8: {
for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);

                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);

                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        l = 0;
                        for (l1 = 1; l1 <= atomic->species.radial_projectors[wakg]; l1++) {

                            ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1 - 1];
                            ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1 - 1];

                            if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 0) {
                                l2 = 0;
                                PFp = 1.0;
                                PFm = 0.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 1) {
                                l2 = 2;
                                PFp = 2.0 / 3.0;
                                PFm = 1.0 / 3.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 2) {
                                l2 = 4;
                                PFp = 3.0 / 5.0;
                                PFm = 2.0 / 5.0;
                            } else if (openmx_atomic_projector_angular(atomic,wakg)[l1] == 3) {
                                l2 = 6;
                                PFp = 4.0 / 7.0;
                                PFm = 3.0 / 7.0;
                            }

                            /* 4 */

                            reference_dHNL_SO_rows(SpinP_switch,&sumx0.r, &sumy0.r, &sumz0.r,
                                &sumx1.r, &sumy1.r, &sumz1.r,
                                &sumx2.r, &sumy2.r, &sumz2.r,
                                &sumx0.i, &sumy0.i, &sumz0.i,
                                &sumx1.i, &sumy1.i, &sumz1.i,
                                &sumx2.i, &sumy2.i, &sumz2.i,
                                -1.0,
                                PFp, PFm,
                                -ene_p, -ene_m,
                                l2, &l,
                                DS_NL1[0][1][Matomnum + 1][kl2][n], DS_NL1[0][2][Matomnum + 1][kl2][n], DS_NL1[0][3][Matomnum + 1][kl2][n],
                                DS_NL1[1][1][Matomnum + 1][kl2][n], DS_NL1[1][2][Matomnum + 1][kl2][n], DS_NL1[1][3][Matomnum + 1][kl2][n],
                                DS_NL1[0][0][Matomnum + 1][kl1][m], DS_NL1[1][0][Matomnum + 1][kl1][m]);
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
break;}
default:abort();}
}
