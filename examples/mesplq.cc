// $Id: mesplq.cc,v 1.2 2002-09-14 19:48:46 edwards Exp $
//
//! Return the value of the average plaquette normalized to 1
/*!
 * \param u -- gauge field (Read)
 * \return w_plaq -- plaquette average (Write)
 * \return s_plaq -- space-like plaquette average (Write)
 * \return t_plaq -- time-like plaquette average (Write)
 */

#include "tests.h"

using namespace QDP;

LatticeReal fred(const LatticeReal& v)
{
  return v;
}


void MesPlq(const multi1d<LatticeGauge>& u, Double& w_plaq, Double& s_plaq, 
	    Double& t_plaq, Double& link)
{
  s_plaq = t_plaq = w_plaq = link = 0.0;

  for(int mu=1; mu < Nd; ++mu)
  {
    for(int nu=0; nu < mu; ++nu)
    {
#if 0
      /* tmp_0 = u(x+mu,nu)*u_dag(x+nu,mu) */
      LatticeGauge tmp_0 = shift(u[nu],FORWARD,mu) * conj(shift(u[mu],FORWARD,nu));

      /* tmp_1 = tmp_0*u_dag(x,nu)=u(x+mu,nu)*u_dag(x+nu,mu)*u_dag(x,nu) */
      LatticeGauge tmp_1 = tmp_0 * conj(u[nu]);

      /* tmp = sum(tr(u(x,mu)*tmp_1=u(x,mu)*u(x+mu,nu)*u_dag(x+nu,mu)*u_dag(x,nu))) */
      Double tmp = sum(real(trace(u[mu]*tmp_1)));

#else
      /* tmp_0 = u(x+mu,nu)*u_dag(x+nu,mu) */
      /* tmp_1 = tmp_0*u_dag(x,nu)=u(x+mu,nu)*u_dag(x+nu,mu)*u_dag(x,nu) */
      /* wplaq_tmp = tr(u(x,mu)*tmp_1=u(x,mu)*u(x+mu,nu)*u_dag(x+nu,mu)*u_dag(x,nu)) */
      Double tmp = real(innerproduct(u[nu],
				     u[mu]*(shift(u[nu],FORWARD,mu)*conj(shift(u[mu],FORWARD,nu)))));
#endif
      w_plaq += tmp;

      if (mu == geom.Tdir() || nu == geom.Tdir())
	t_plaq += tmp;
      else 
	s_plaq += tmp;
    }
  }
  
  w_plaq *= 2.0 / double(geom.Vol()*Nd*(Nd-1)*Nc);
  
  if (Nd > 2) 
    s_plaq *= 2.0 / double(geom.Vol()*(Nd-1)*(Nd-2)*Nc);
  
  t_plaq = t_plaq / double(geom.Vol()*(Nd-1)*Nc);
  
  for(int mu=0; mu < Nd; ++mu)
  { 
//    LatticeReal link_tmp = trace_real(u[mu]);
//    link += sum(link_tmp);

//    link += sum(trace_real(u[mu]));
    link += sum(LatticeReal(real(trace(u[mu]))));

//    link += sum(fred(trace_real(u[mu])));
  }

  link = link / double(geom.Vol()*Nd*Nc);
}
