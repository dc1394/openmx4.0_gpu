#include <openmx/basis/density_table.hpp>
#include <openmx/basis/density.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
// Kumogata interpolation and radial derivative, T. Ozaki (2012).
// Preserve each function's distinct coordinate arithmetic and endpoint rule.
double density_value_unchecked(int N, double x, const double *xv, const double *rv, const double *yv)
{

  if (x<xv[0]){

    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,f,df,r;
    double g1,g2,x1,x2,y1,y2,y12,y22,a,b;

    r = std::exp(x);

    m = 4;
    rm = rv[m];

    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];

    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;

    a = 0.5*df/rm;
    b = f - a*rm*rm;
    return a*r*r + b;

  }

  else{

    int i;
    double t,dt;
    double xmin,xmax;

    xmin = xv[0];
    xmax = xv[N-1];
    x = x<xmax ? x : xmax;
    x = x>xmin ? x : xmin;
    t = ((double)N-1.0)*(x-xmin)/(xmax-xmin);
    /* The table has N+2 entries. The endpoint needs no yv[N+2].
       Rounding can reach it even when x is just below xmax. */
    if (t >= (double)N-1.0) return yv[N];
    i = std::floor(t);
    dt = t - (double)i;

    return 0.5*( ((yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
		  -yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i])*dt
		 +(yv[i+2]-yv[i]))*dt
                 +yv[i+1];
  }

}
double density_derivative_unchecked(int N, double x, double r, const double *xv, const double *rv, const double *yv)
{

  if (x<xv[0]){

    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,a,b,r;
    double g1,g2,x1,x2,y1,y2,y12,y22,f,df;

    r = std::exp(x);

    m = 4;
    rm = rv[m];

    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];

    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;

    a = 0.5*df/rm;
    b = f - a*rm*rm;
    (void)b;
    return 2.0*a*r;
  }

  else{

    int i;
    double t,dt;
    double xmin,xmax,tmp;

    xmin = xv[0];
    xmax = xv[N-1];
    x = x<xmax ? x : xmax;
    x = x>xmin ? x : xmin;

    tmp = ((double)N-1.0)/(xmax-xmin);
    t = (x-xmin)*tmp;
    /* The table has N+2 entries. The endpoint needs no yv[N+2].
       Rounding can reach it even when x is just below xmax. */
    if (t >= (double)N-1.0) return 0.5*(yv[N+1]-yv[N-1])*tmp/r;
    i = std::floor(t);
    dt = t - (double)i;

    return 0.5*(( 3.0*(yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
		  +2.0*(-yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i]))*dt
		  +(yv[i+2]-yv[i]))*tmp/r;
  }

}

bool valid_rows(int count,double x,const double* log_grid,const double* radii,const double* values)
{
  return count>=6 && count<=INT_MAX-2 && log_grid && radii && values && !std::isnan(x);
}
void validate_radius(double r)
{
  if(!std::isfinite(r) || r<0)throw std::invalid_argument("invalid density radius");
}
} // namespace

extern "C" double openmx_basis_density_value(int count,double x,const double* log_grid,
    const double* radii,const double* values)
{
  if(!valid_rows(count,x,log_grid,radii,values))return std::numeric_limits<double>::quiet_NaN();
  return density_value_unchecked(count,x,log_grid,radii,values);
}
extern "C" double openmx_basis_density_derivative(int count,double x,double radius,
    const double* log_grid,const double* radii,const double* values)
{
  if(!valid_rows(count,x,log_grid,radii,values) ||
     (x>=log_grid[0] && (!std::isfinite(radius) || radius<=0)))
    return std::numeric_limits<double>::quiet_NaN();
  return density_derivative_unchecked(count,x,radius,log_grid,radii,values);
}

namespace openmx::basis {
LogDensityView::LogDensityView(std::span<const double> log_grid,std::span<const double> radii,
    std::span<const double> padded_density)
  :LogDensityView(log_grid,radii,padded_density,Validated{})
{
  const auto n=log_grid_.size();
  if(n<6 || n>INT_MAX-2 || radii_.size()!=n || values_.size()!=n+2)
    throw std::invalid_argument("invalid log-density table shape");
  for(std::size_t i=0;i<n;++i){
    if(!std::isfinite(log_grid_[i]) || !std::isfinite(radii_[i]) || radii_[i]<=0 ||
       (i && (log_grid_[i]<=log_grid_[i-1] || radii_[i]<=radii_[i-1])))
      throw std::invalid_argument("invalid log-density coordinate");
  }
  for(std::size_t i=0;i<n;++i){
    const double expected=log_grid_.front()+static_cast<double>(i)*(log_grid_.back()-log_grid_.front())/static_cast<double>(n-1);
    if(std::abs(log_grid_[i]-expected)>1e-10*(1+std::abs(expected)) ||
       std::abs(std::log(radii_[i])-log_grid_[i])>1e-10*(1+std::abs(log_grid_[i])))
      throw std::invalid_argument("density mesh must be uniform in log radius and consistent with radii");
  }
  for(double value:values_)if(!std::isfinite(value))throw std::invalid_argument("non-finite density value");
}

double LogDensityView::sample_log(double x) const
{
  if(std::isnan(x))throw std::invalid_argument("invalid logarithmic density coordinate");
  return density_value_unchecked(static_cast<int>(log_grid_.size()),x,log_grid_.data(),radii_.data(),values_.data());
}
double LogDensityView::sample(double radius) const
{
  validate_radius(radius);
  const double x=radius==0 ? -std::numeric_limits<double>::infinity() : std::log(radius);
  return sample_log(x);
}
double LogDensityView::derivative(double radius) const
{
  validate_radius(radius);
  const double x=radius==0 ? -std::numeric_limits<double>::infinity() : std::log(radius);
  return density_derivative_unchecked(static_cast<int>(log_grid_.size()),x,radius,
      log_grid_.data(),radii_.data(),values_.data());
}

LogDensityTable::LogDensityTable(std::span<const double> log_grid,std::span<const double> radii,
    std::span<const double> physical_density)
{
  const auto n=log_grid.size();
  if(n<6 || n>INT_MAX-2 || radii.size()!=n || physical_density.size()!=n)
    throw std::invalid_argument("invalid physical density shape");
  log_grid_.assign(log_grid.begin(),log_grid.end());radii_.assign(radii.begin(),radii.end());
  values_.resize(n+2);std::copy(physical_density.begin(),physical_density.end(),values_.begin()+1);
  values_[0]=2.0*values_[1]-values_[2];values_[n+1]=2.0*values_[n]-values_[n-1];
  (void)LogDensityView(log_grid_,radii_,values_);
}
LogDensityView LogDensityTable::view() const
{
  if(log_grid_.size()<6)throw std::logic_error("moved-from density table");
  return {log_grid_,radii_,values_,LogDensityView::Validated{}};
}

double atomic_hartree_potential(const LogDensityView& density,double R,const GaussLegendreView& quadrature)
{
  if(!std::isfinite(R) || R<=0)throw std::invalid_argument("atomic Hartree radius must be positive");
  constexpr double pi=3.1415926535897932384626;
  const auto log_grid=density.log_grid(),radii=density.radii(),values=density.padded_density();
  const auto nodes=quadrature.nodes(),weights=quadrature.weights();
  const int count=static_cast<int>(log_grid.size());
  double xmin=log_grid.front(),xmax=std::log(R),Sx=xmax+xmin,Dx=xmax-xmin;
  double inside=0.;
  for(std::size_t i=0;i<nodes.size();++i){
    const double x=0.50*(Dx*nodes[i]+Sx),rp=std::exp(x);
    inside=inside+density_value_unchecked(count,x,log_grid.data(),radii.data(),values.data())*weights[i]*rp*rp*rp;
  }
  inside=0.50*Dx*inside;
  inside=4.0*pi*inside/R;
  xmin=std::log(R);xmax=log_grid.back();Sx=xmax+xmin;Dx=xmax-xmin;
  double outside=0.;
  for(std::size_t i=0;i<nodes.size();++i){
    const double x=0.50*(Dx*nodes[i]+Sx),rp=std::exp(x);
    outside=outside+density_value_unchecked(count,x,log_grid.data(),radii.data(),values.data())*weights[i]*rp*rp;
  }
  outside=0.50*Dx*outside;
  outside=4.0*pi*outside;
  const double result=inside+outside;
  if(!std::isfinite(result))throw std::overflow_error("non-finite atomic Hartree potential");
  return result;
}
} // namespace openmx::basis
