/* Frozen core-hole spinor derivative from 0216ce1d; independent legacy reference. */
static void reference_dHCH_SO(double* sumx0r, double* sumx0i, double* sumy0r, double* sumy0i, double* sumz0r, double* sumz0i,
    double* sumx1r, double* sumx1i, double* sumy1r, double* sumy1i, double* sumz1r, double* sumz1i,
    double* sumx2r, double* sumx2i, double* sumy2r, double* sumy2i, double* sumz2r, double* sumz2i,
    double fugou,
    int Mc_AN, int k, int m,
    int Mj_AN, int kl, int n,
    int kg, int wakg,
    double penalty_value,
    double***** OLP1)
{
    int L, L2, l, mul, apply_flag;
    double d12_m12, d12_p12;
    double d32_m12, d32_p12, d32_m32, d32_p32;
    double d52_m12, d52_p12, d52_m32, d52_p32, d52_m52, d52_p52;
    double d72_m12, d72_p12, d72_m32, d72_p32, d72_m52, d72_p52, d72_m72, d72_p72;

    if (kg != Core_Hole_Atom)
        return;

    /****************************************************
                   set penalty coefficients
    ****************************************************/

    L = 0;
    for (l = 0; l <= Spe_MaxL_Basis[wakg]; l++) {
        for (mul = 0; mul < Spe_Num_Basis[wakg][l]; mul++) {

            apply_flag = 0;

            if ((strcmp(Core_Hole_Orbital, "s") == 0) && l == 0 && mul == 0) {

                L2 = 0;
                apply_flag = 1;
            }

            else if ((strcmp(Core_Hole_Orbital, "p") == 0) && l == 1 && mul == 0) {

                L2 = 2;
                apply_flag = 1;
            }

            else if ((strcmp(Core_Hole_Orbital, "d") == 0) && l == 2 && mul == 0) {

                L2 = 4;
                apply_flag = 1;
            }

            else if ((strcmp(Core_Hole_Orbital, "f") == 0) && l == 3 && mul == 0) {

                L2 = 6;
                apply_flag = 1;
            }

            if (apply_flag == 1) {

                /****************************************************
                         set coefficients related to penalty
                ****************************************************/

                if (L2 == 0) {

                    if (Core_Hole_J == 0) {
                        /* for Core_Hole_J==0 */
                        d12_p12 = penalty_value;
                        d12_m12 = penalty_value;
                    }

                    else {
                        /* for Core_Hole_J!=0 */
                        d12_p12 = penalty_value * (Core_Hole_J == 1);
                        d12_m12 = penalty_value * (Core_Hole_J == 2);
                    }
                }

                else if (L2 == 2) {

                    if (Core_Hole_J == 0) {
                        /* for Core_Hole_J==0 */
                        d32_p32 = penalty_value / 3.0;
                        d32_p12 = penalty_value / 3.0;
                        d32_m12 = penalty_value / 3.0;
                        d32_m32 = penalty_value / 3.0;

                        d12_p12 = penalty_value / 3.0;
                        d12_m12 = penalty_value / 3.0;
                    }

                    else {
                        /* for Core_Hole_J!=0 */
                        d32_p32 = penalty_value / 3.0 * (Core_Hole_J == 1);
                        d32_p12 = penalty_value / 3.0 * (Core_Hole_J == 2);
                        d32_m12 = penalty_value / 3.0 * (Core_Hole_J == 3);
                        d32_m32 = penalty_value / 3.0 * (Core_Hole_J == 4);

                        d12_p12 = penalty_value / 3.0 * (Core_Hole_J == 5);
                        d12_m12 = penalty_value / 3.0 * (Core_Hole_J == 6);
                    }
                }

                else if (L2 == 4) {

                    if (Core_Hole_J == 0) {
                        /* for Core_Hole_J==0 */
                        d52_p52 = penalty_value / 5.0;
                        d52_p32 = penalty_value / 5.0;
                        d52_p12 = penalty_value / 5.0;
                        d52_m12 = penalty_value / 5.0;
                        d52_m32 = penalty_value / 5.0;
                        d52_m52 = penalty_value / 5.0;

                        d32_p32 = penalty_value / 5.0;
                        d32_p12 = penalty_value / 5.0;
                        d32_m12 = penalty_value / 5.0;
                        d32_m32 = penalty_value / 5.0;
                    }

                    else {
                        /* for Core_Hole_J!=0 */
                        d52_p52 = penalty_value / 5.0 * (Core_Hole_J == 1);
                        d52_p32 = penalty_value / 5.0 * (Core_Hole_J == 2);
                        d52_p12 = penalty_value / 5.0 * (Core_Hole_J == 3);
                        d52_m12 = penalty_value / 5.0 * (Core_Hole_J == 4);
                        d52_m32 = penalty_value / 5.0 * (Core_Hole_J == 5);
                        d52_m52 = penalty_value / 5.0 * (Core_Hole_J == 6);

                        d32_p32 = penalty_value / 5.0 * (Core_Hole_J == 7);
                        d32_p12 = penalty_value / 5.0 * (Core_Hole_J == 8);
                        d32_m12 = penalty_value / 5.0 * (Core_Hole_J == 9);
                        d32_m32 = penalty_value / 5.0 * (Core_Hole_J == 10);
                    }
                }

                else if (L2 == 6) {

                    if (Core_Hole_J == 0) {
                        /* for Core_Hole_J==0 */
                        d72_p72 = penalty_value / 7.0;
                        d72_p52 = penalty_value / 7.0;
                        d72_p32 = penalty_value / 7.0;
                        d72_p12 = penalty_value / 7.0;
                        d72_m12 = penalty_value / 7.0;
                        d72_m32 = penalty_value / 7.0;
                        d72_m52 = penalty_value / 7.0;
                        d72_m72 = penalty_value / 7.0;

                        d52_p52 = penalty_value / 7.0;
                        d52_p32 = penalty_value / 7.0;
                        d52_p12 = penalty_value / 7.0;
                        d52_m12 = penalty_value / 7.0;
                        d52_m32 = penalty_value / 7.0;
                        d52_m52 = penalty_value / 7.0;
                    }

                    else {
                        /* for Core_Hole_J!=0 */
                        d72_p72 = penalty_value / 7.0 * (Core_Hole_J == 1);
                        d72_p52 = penalty_value / 7.0 * (Core_Hole_J == 2);
                        d72_p32 = penalty_value / 7.0 * (Core_Hole_J == 3);
                        d72_p12 = penalty_value / 7.0 * (Core_Hole_J == 4);
                        d72_m12 = penalty_value / 7.0 * (Core_Hole_J == 5);
                        d72_m32 = penalty_value / 7.0 * (Core_Hole_J == 6);
                        d72_m52 = penalty_value / 7.0 * (Core_Hole_J == 7);
                        d72_m72 = penalty_value / 7.0 * (Core_Hole_J == 8);

                        d52_p52 = penalty_value / 7.0 * (Core_Hole_J == 9);
                        d52_p32 = penalty_value / 7.0 * (Core_Hole_J == 10);
                        d52_p12 = penalty_value / 7.0 * (Core_Hole_J == 11);
                        d52_m12 = penalty_value / 7.0 * (Core_Hole_J == 12);
                        d52_m32 = penalty_value / 7.0 * (Core_Hole_J == 13);
                        d52_m52 = penalty_value / 7.0 * (Core_Hole_J == 14);
                    }
                }

                /****************************************************
                          off-diagonal contribution on up-dn
                             for spin non-collinear
                ****************************************************/

                if (SpinP_switch == 3) {

                    /***************
                           p
                    ***************/

                    if (L2 == 2) {

                        /* real contribution of l+1/2 to off-diagonal up-down matrix */
                        *sumx2r += fugou * (d32_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d32_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        *sumy2r += fugou * (d32_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d32_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        *sumz2r += fugou * (d32_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d32_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */
                        *sumx2i += fugou * (-d32_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d32_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        *sumx2i += fugou * (-d32_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d32_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        *sumx2i += fugou * (-d32_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d32_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        /* real contribution of l-1/2 for to off-diagonal up-down matrix */
                        *sumx2r -= fugou * (d12_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d12_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        *sumy2r -= fugou * (d12_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d12_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        *sumz2r -= fugou * (d12_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - d12_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L]);

                        /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */
                        *sumx2i -= fugou * (-d12_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d12_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        *sumy2i -= fugou * (-d12_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d12_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);

                        *sumz2i -= fugou * (-d12_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + d12_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1]);
                    }

                    /***************
                           d
                    ***************/

                    else if (L2 == 4) {

                        /* real contribution of l+1/2 to off diagonal up-down matrix */

                        *sumx2r += fugou * (-sqrt(3.0) * d52_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d52_m12 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d52_p32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d52_m32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumy2r += fugou * (-sqrt(3.0) * d52_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d52_m12 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d52_p32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d52_m32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumz2r += fugou * (-sqrt(3.0) * d52_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d52_m12 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d52_p32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d52_m32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

                        *sumx2i += fugou * (sqrt(3.0) * d52_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d52_m12 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d52_m32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d52_p32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumy2i += fugou * (sqrt(3.0) * d52_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d52_m12 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d52_m32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d52_p32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumz2i += fugou * (sqrt(3.0) * d52_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d52_m12 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d52_m32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d52_p32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d52_m32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d52_p32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        /* real contribution of l-1/2 for to diagonal up-down matrix */

                        *sumx2r -= fugou * (-sqrt(3.0) * d32_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d32_m12 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d32_p32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d32_m32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumy2r -= fugou * (-sqrt(3.0) * d32_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d32_m12 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d32_p32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d32_m32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumz2r -= fugou * (-sqrt(3.0) * d32_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(3.0) * d32_m12 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] - d32_p32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] + d32_m32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

                        *sumx2i -= fugou * (sqrt(3.0) * d32_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d32_m12 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d32_m32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d32_p32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumy2i -= fugou * (sqrt(3.0) * d32_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d32_m12 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d32_m32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d32_p32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);

                        *sumz2i -= fugou * (sqrt(3.0) * d32_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(3.0) * d32_m12 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L] + d32_m32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - d32_p32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - d32_m32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + d32_p32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2]);
                    }

                    /***************
                                   f
                    ***************/

                    else if (L2 == 6) {

                        /* real contribution of l+1/2 to off diagonal up-down matrix */

                        *sumx2r += fugou * (-sqrt(6.0) * d72_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d72_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d72_p32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d72_m32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d72_p52 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d72_m52 * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumy2r += fugou * (-sqrt(6.0) * d72_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d72_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d72_p32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d72_m32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d72_p52 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d72_m52 * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumz2r += fugou * (-sqrt(6.0) * d72_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d72_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d72_p32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d72_m32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d72_p52 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d72_m52 * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

                        *sumx2i += fugou * (sqrt(6.0) * d72_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d72_m12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d72_p32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d72_m32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d72_p52 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d72_m52 * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumy2i += fugou * (sqrt(6.0) * d72_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d72_m12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d72_p32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d72_m32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d72_p52 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d72_m52 * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumz2i += fugou * (sqrt(6.0) * d72_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d72_m12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d72_p32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d72_m32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d72_p32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d72_m32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d72_p52 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d72_m52 * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d72_p52 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d72_m52 * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        /* real contribution of l-1/2 for to off-diagonal up-down matrix */

                        *sumx2r -= fugou * (-sqrt(6.0) * d52_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d52_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d52_p32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d52_m32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d52_p52 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d52_m52 * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumy2r -= fugou * (-sqrt(6.0) * d52_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d52_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d52_p32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d52_m32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d52_p52 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d52_m52 * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumz2r -= fugou * (-sqrt(6.0) * d52_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + sqrt(6.0) * d52_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L] - sqrt(2.5) * d52_p32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 4] + sqrt(2.5) * d52_m32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(1.5) * d52_p52 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 6] + sqrt(1.5) * d52_m52 * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

                        *sumx2i -= fugou * (sqrt(6.0) * d52_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d52_m12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d52_p32 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d52_m32 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d52_p52 * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d52_m52 * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumy2i -= fugou * (sqrt(6.0) * d52_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d52_m12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d52_p32 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d52_m32 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d52_p52 * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d52_m52 * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);

                        *sumz2i -= fugou * (sqrt(6.0) * d52_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 2] - sqrt(6.0) * d52_m12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L] + sqrt(2.5) * d52_p32 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 4] - sqrt(2.5) * d52_m32 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 1] - sqrt(2.5) * d52_p32 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 3] + sqrt(2.5) * d52_m32 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 2] + sqrt(1.5) * d52_p52 * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 6] - sqrt(1.5) * d52_m52 * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 3] - sqrt(1.5) * d52_p52 * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 5] + sqrt(1.5) * d52_m52 * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 4]);
                    }

                } /* if (SpinP_switch==3) */

                /****************************************************
                    off-diagonal contribution on up-up and dn-dn
                ****************************************************/

                /* p */

                if (L2 == 2) {

                    /* contribution of l+1/2 for up spin */

                    *sumx0i += 0.5 * fugou * (+(d32_m12 - 3.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (-d32_m12 + 3.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumy0i += 0.5 * fugou * (+(d32_m12 - 3.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (-d32_m12 + 3.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumz0i += 0.5 * fugou * (+(d32_m12 - 3.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (-d32_m12 + 3.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    /* contribution of l+1/2 for down spin */

                    *sumx1i += 0.5 * fugou * (+(-d32_p12 + 3.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (+d32_p12 - 3.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumy1i += 0.5 * fugou * (+(-d32_p12 + 3.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (+d32_p12 - 3.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumz1i += 0.5 * fugou * (+(-d32_p12 + 3.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + (+d32_p12 - 3.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    /* contribution of l-1/2 for up spin */

                    *sumx0i += fugou * (d12_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] - d12_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumy0i += fugou * (d12_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] - d12_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumz0i += fugou * (d12_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] - d12_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    /* contribution of l-1/2 for down spin */

                    *sumx1i += fugou * (-d12_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + d12_p12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumy1i += fugou * (-d12_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + d12_p12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);

                    *sumz1i += fugou * (-d12_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L + 1] + d12_p12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L]);
                }

                /* d */

                else if (L2 == 4) {

                    /* contribution of l+1/2 for up spin */

                    *sumx0i += 0.5 * fugou * (+(d52_m32 - 5.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-d52_m32 + 5.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d52_m12 - 4.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumy0i += 0.5 * fugou * (+(d52_m32 - 5.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-d52_m32 + 5.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d52_m12 - 4.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumz0i += 0.5 * fugou * (+(d52_m32 - 5.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-d52_m32 + 5.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d52_m12 - 4.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    /* contribution of l+1/2 for down spin */

                    *sumx1i += 0.5 * fugou * (-(d52_p32 - 5.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-d52_p32 + 5.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d52_p12 - 4.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumy1i += 0.5 * fugou * (-(d52_p32 - 5.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-d52_p32 + 5.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d52_p12 - 4.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumz1i += 0.5 * fugou * (-(d52_p32 - 5.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-d52_p32 + 5.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d52_p12 - 4.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    /* contribution of l-1/2 for up spin */

                    *sumx0i += 0.5 * fugou * ((4.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (3.0 * d32_m12 - d32_p32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-3.0 * d32_m12 + d32_p32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumy0i += 0.5 * fugou * ((4.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (3.0 * d32_m12 - d32_p32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-3.0 * d32_m12 + d32_p32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumz0i += 0.5 * fugou * ((4.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (3.0 * d32_m12 - d32_p32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-3.0 * d32_m12 + d32_p32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    /* contribution of l-1/2 for down spin */

                    *sumx1i += 0.5 * fugou * ((-4.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (+4.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (3.0 * d32_p12 - d32_m32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-3.0 * d32_p12 + d32_m32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumy1i += 0.5 * fugou * ((-4.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (+4.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (3.0 * d32_p12 - d32_m32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-3.0 * d32_p12 + d32_m32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);

                    *sumz1i += 0.5 * fugou * ((-4.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (+4.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (3.0 * d32_p12 - d32_m32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-3.0 * d32_p12 + d32_m32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3]);
                }

                /* f */

                else if (L2 == 6) {

                    /* contribution of l+1/2 for up spin */

                    *sumx0i += 0.5 * fugou * ((3.0 * d72_m12 - 5.0 * d72_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d72_m32 - 6.0 * d72_p52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (1.0 * d72_m52 - 7.0 * d72_p72) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumy0i += 0.5 * fugou * ((3.0 * d72_m12 - 5.0 * d72_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d72_m32 - 6.0 * d72_p52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (1.0 * d72_m52 - 7.0 * d72_p72) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumz0i += 0.5 * fugou * ((3.0 * d72_m12 - 5.0 * d72_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (2.0 * d72_m32 - 6.0 * d72_p52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (1.0 * d72_m52 - 7.0 * d72_p72) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    /* contribution of l+1/2 for down spin */

                    *sumx1i += 0.5 * fugou * (-(3.0 * d72_p12 - 5.0 * d72_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d72_p32 - 6.0 * d72_m52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (1.0 * d72_p52 - 7.0 * d72_m72) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumy1i += 0.5 * fugou * (-(3.0 * d72_p12 - 5.0 * d72_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d72_p32 - 6.0 * d72_m52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (1.0 * d72_p52 - 7.0 * d72_m72) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumz1i += 0.5 * fugou * (-(3.0 * d72_p12 - 5.0 * d72_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (2.0 * d72_p32 - 6.0 * d72_m52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (1.0 * d72_p52 - 7.0 * d72_m72) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    /* contribution of l-1/2 for up spin */

                    *sumx0i += 0.5 * fugou * ((4.0 * d52_m12 - 2.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (5.0 * d52_m32 - 1.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (+6.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-6.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumy0i += 0.5 * fugou * ((4.0 * d52_m12 - 2.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (5.0 * d52_m32 - 1.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (+6.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-6.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumz0i += 0.5 * fugou * ((4.0 * d52_m12 - 2.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] + (-4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] + (5.0 * d52_m32 - 1.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] + (-5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] + (+6.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] + (-6.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    /* contribution of l-1/2 for down spin */

                    *sumx1i += 0.5 * fugou * (-(4.0 * d52_p12 - 2.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (5.0 * d52_p32 - 1.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (+6.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-6.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumy1i += 0.5 * fugou * (-(4.0 * d52_p12 - 2.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (5.0 * d52_p32 - 1.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (+6.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-6.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);

                    *sumz1i += 0.5 * fugou * (-(4.0 * d52_p12 - 2.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 2] - (-4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 1] - (5.0 * d52_p32 - 1.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 4] - (-5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 3] - (+6.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 6] - (-6.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 5]);
                }

                /****************************************************
                     diagonal contribution on up-up and dn-dn
                ****************************************************/

                /* s */

                if (L2 == 0) {

                    /* VNL for j=l+1/2 */

                    *sumx0r += d12_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += d12_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += d12_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];

                    *sumx1r += d12_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += d12_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += d12_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];

                    /* note that VNL for j=l-1/2 is zero */
                }

                /* p */

                else if (L2 == 2) {

                    /* VNL for j=l+1/2 */
                    *sumx0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (4.0 * d32_p12) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumy0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (4.0 * d32_p12) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumz0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (d32_m12 + 3.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (4.0 * d32_p12) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumx1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (4.0 * d32_m12) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumy1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (4.0 * d32_m12) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumz1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (d32_p12 + 3.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (4.0 * d32_m12) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    /* VNL for j=l-1/2 */
                    *sumx0r += d12_m12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += d12_m12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += d12_p12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumy0r += d12_m12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += d12_m12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += d12_p12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumz0r += d12_m12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += d12_m12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += d12_p12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumx1r += d12_p12 * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += d12_p12 * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += d12_m12 * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumy1r += d12_p12 * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += d12_p12 * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += d12_m12 * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];

                    *sumz1r += d12_p12 * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += d12_p12 * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += d12_m12 * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                }

                /* d */

                else if (L2 == 4) {

                    /* VNL for j=l+1/2 */
                    *sumx0r += 0.5 * (6.0 * d52_p12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumy0r += 0.5 * (6.0 * d52_p12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumz0r += 0.5 * (6.0 * d52_p12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz0r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumx1r += 0.5 * (6.0 * d52_m12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumy1r += 0.5 * (6.0 * d52_m12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumz1r += 0.5 * (6.0 * d52_m12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz1r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    /* VNL for j=l-1/2 */
                    *sumx0r += 0.5 * (4.0 * d32_p12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (4.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (4.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumy0r += 0.5 * (4.0 * d32_p12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (4.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (4.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumz0r += 0.5 * (4.0 * d32_p12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (4.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (4.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz0r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumx1r += 0.5 * (4.0 * d32_m12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (4.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (4.0 * d32_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumy1r += 0.5 * (4.0 * d32_m12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (4.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (4.0 * d32_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];

                    *sumz1r += 0.5 * (4.0 * d32_m12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (4.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (4.0 * d32_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz1r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                }

                /* f */

                else if (L2 == 6) {

                    /* VNL for j=l+1/2 */
                    *sumx0r += 0.5 * (8.0 * d72_p12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumx0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumx0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumy0r += 0.5 * (8.0 * d72_p12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumy0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumy0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumz0r += 0.5 * (8.0 * d72_p12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz0r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumz0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumz0r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumx1r += 0.5 * (8.0 * d72_m12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumx1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumx1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumy1r += 0.5 * (8.0 * d72_m12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumy1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumy1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumz1r += 0.5 * (8.0 * d72_m12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz1r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumz1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumz1r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    /* VNL for j=l-1/2 */
                    *sumx0r += 0.5 * (6.0 * d52_p12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumx0r += 0.5 * (6.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumx0r += 0.5 * (6.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumy0r += 0.5 * (6.0 * d52_p12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumy0r += 0.5 * (6.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumy0r += 0.5 * (6.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumz0r += 0.5 * (6.0 * d52_p12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz0r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz0r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumz0r += 0.5 * (6.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumz0r += 0.5 * (6.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumx1r += 0.5 * (6.0 * d52_m12) * OLP1[1][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumx1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumx1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[1][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumx1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumx1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[1][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumx1r += 0.5 * (6.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumx1r += 0.5 * (6.0 * d52_p52) * OLP1[1][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumy1r += 0.5 * (6.0 * d52_m12) * OLP1[2][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumy1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumy1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[2][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumy1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumy1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[2][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumy1r += 0.5 * (6.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumy1r += 0.5 * (6.0 * d52_p52) * OLP1[2][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];

                    *sumz1r += 0.5 * (6.0 * d52_m12) * OLP1[3][Mc_AN][k][m][L] * OLP1[0][Mj_AN][kl][n][L];
                    *sumz1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 1] * OLP1[0][Mj_AN][kl][n][L + 1];
                    *sumz1r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * OLP1[3][Mc_AN][k][m][L + 2] * OLP1[0][Mj_AN][kl][n][L + 2];
                    *sumz1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 3] * OLP1[0][Mj_AN][kl][n][L + 3];
                    *sumz1r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * OLP1[3][Mc_AN][k][m][L + 4] * OLP1[0][Mj_AN][kl][n][L + 4];
                    *sumz1r += 0.5 * (6.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 5] * OLP1[0][Mj_AN][kl][n][L + 5];
                    *sumz1r += 0.5 * (6.0 * d52_p52) * OLP1[3][Mc_AN][k][m][L + 6] * OLP1[0][Mj_AN][kl][n][L + 6];
                }

            } /* if (apply_flag==1) */

            /* increment of L */

            L += 2 * l + 1;

        } /* mul */
    } /* l */
}
