/* The two application symbols the frozen XC routines need. */
#include "openmx_common.h"

int dc_Type = 0;

double largest(double a, double b)
{
  double result;

  if (b<=a) result = a;
  else      result = b;
  return result;
}
