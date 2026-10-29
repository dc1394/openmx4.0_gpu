/* Frozen orbital radial formulas from Get_dOrbitals.c at
   fdada050f4c8c3322c5483a4dfe291103a5d4a2a. Only data access/loop wrappers
   are adapted to one row; numeric expressions are the original C.
   T. Ozaki, 22 November 2001. See ../../LICENSE. */
#include <stddef.h>
void reference_orbital_radial(int angular,const double *radii,const double *row,
    size_t points,double R,double out[2])
{
  int mp_min,mp_max,m,po;
  double rm,h1,h2,h3,f1,f2,f3,f4,dum,dum1,dum2,dum3,dum4;
  double g1,g2,x1,x2,y1,y2,y12,y22,f,df,a,b,c,d;
  po = 0;
  mp_min = 0;
  mp_max = ((int)points) - 1;

  if (radii[((int)points)-1]<R){

    { const int L0=angular;
      {
         out[0] = 0.0;
        out[1] = 0.0;
      }
    }

    po = 1;
  }

  else if (R<radii[0]){

    m = 4;
    rm = radii[m];

    /* Use reflected endpoint samples before forming the spline weights. */
    h1 = (m==1) ? radii[m-1] - radii[m+1]
      : radii[m-1] - radii[m-2];
    h2 = radii[m]   - radii[m-1];
    h3 = (m==((int)points)-1) ? radii[m-2] - radii[m]
      : radii[m+1] - radii[m];

    x1 = rm - radii[m-1];
    x2 = rm - radii[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    dum = h1 + h2;
    dum1 = h1/h2/dum;
    dum2 = h2/h1/dum;
    dum = h2 + h3;
    dum3 = h2/h3/dum;
    dum4 = h3/h2/dum;

    { const int L0=angular;
      {

        f1 = row[(m==1) ? m+1 : m-2];
        f2 = row[m-1];
        f3 = row[m];
        f4 = row[(m==((int)points)-1) ? m-2 : m+1];

        dum = f3 - f2;
        g1 = dum*dum1 + (f2-f1)*dum2;
        g2 = (f4-f3)*dum3 + dum*dum4;

        f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

        df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y22*(2.0*f2 + h2*g1)/h2
           + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
           - y12*(2.0*f3 - h2*g2)/h2;

        if (L0==0){
          a = 0.0;
          b = 0.5*df/rm;
          c = 0.0;
          d = f - b*rm*rm;
        }

        else if (L0==1){
          a = (rm*df - f)/(2.0*rm*rm*rm);
          b = 0.0;
          c = df - 3.0*a*rm*rm;
          d = 0.0;
        }

        else{
          b = (3.0*f - rm*df)/(rm*rm);
          a = (f - b*rm*rm)/(rm*rm*rm);
          c = 0.0;
          d = 0.0;
        }

        out[0] = a*R*R*R + b*R*R + c*R + d;
        out[1] = 3.0*a*R*R + 2.0*b*R + c;

      }
    }

  }

  else{

    do{
      m = (mp_min + mp_max)/2;
      if (radii[m]<R)
        mp_min = m;
      else
        mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;

    h1 = (m==1) ? radii[m-1] - radii[m+1]
      : radii[m-1] - radii[m-2];
    h2 = radii[m]   - radii[m-1];
    h3 = (m==((int)points)-1) ? radii[m-2] - radii[m]
      : radii[m+1] - radii[m];

    x1 = R - radii[m-1];
    x2 = R - radii[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    dum = h1 + h2;
    dum1 = h1/h2/dum;
    dum2 = h2/h1/dum;
    dum = h2 + h3;
    dum3 = h2/h3/dum;
    dum4 = h3/h2/dum;

    { const int L0=angular;
      {

        f1 = row[(m==1) ? m+1 : m-2];
        f2 = row[m-1];
        f3 = row[m];
        f4 = row[(m==((int)points)-1) ? m-2 : m+1];

        dum = f3 - f2;
        g1 = dum*dum1 + (f2-f1)*dum2;
        g2 = (f4-f3)*dum3 + dum*dum4;

        f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

        df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y2*y2*(2.0*f2 + h2*g1)/h2
           + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
           - y1*y1*(2.0*f3 - h2*g2)/h2;

         out[0] = f;
        out[1] = df;

      }
    }
  }

  /* dr/dx,y,z, dQ/dx,y,z, dP/dx,y,z and dAngular */


  (void)po;
}
