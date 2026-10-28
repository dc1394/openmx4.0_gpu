/* What the frozen OpenMX 4.0 XC routines in this directory take from the
   application's openmx_common.h: the PI literal, the largest() helper, the
   LDA+U flag dc_Type and their own prototypes.  The routines themselves are
   the committed sources, unmodified. */
#ifndef OPENMX_XC_FIXTURE_COMMON_H
#define OPENMX_XC_FIXTURE_COMMON_H

#define PI 3.1415926535897932384626

extern int dc_Type;
double largest(double a, double b);

double XC_Ceperly_Alder(double den, int P_switch);
void XC_CA_LSDA(int SCF_iter, double den0, double den1, double XC[2], double X[2], double C[2], int P_switch);
void XC_PW92C(int SCF_iter, double dens[2], double Ec[1], double Vc[2]);
void XC_PBE(int SCF_iter, double dens[2], double GDENS[3][2], double Exc[2],
            double DEXDD[2], double DECDD[2],
            double DEXDGD[3][2], double DECDGD[3][2]);
void XC_EX(int NSP, double DS0, double DS[2], double EX[1], double VX[2]);

#endif
