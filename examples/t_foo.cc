// -*- C++ -*-
//
// $Id: t_foo.cc,v 1.39 2004-07-12 16:33:28 edwards Exp $
//
/*! \file
 *  \brief Silly little internal test code
 */


#include "qdp.h"
#include "qdp_iogauge.h"

//using namespace std;

using namespace QDP;



int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {2,1,1,1};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

  XMLFileWriter xml("t_foo.xml");
//  xml.open("foo.xml");
  push(xml, "foo");

  write(xml,"nrow", nrow);
  write(xml,"logicalSize",Layout::logicalSize());

#if 0
  {
    LatticeColorMatrix a,b,c;
    LatticeComplex cc, dd;
    random(a); random(b); random(c);

    c = a*b;
    cc = trace(c);
    cerr << "Here 1" << endl;
    dd = trace(a*b);
    cerr << "Here 2" << endl;
    dd = trace(a*(b*1));
    cerr << "Here 3" << endl;
    write(xml,"diff", Real(norm2(cc-dd)));

    c = adj(a)*b;
    cc = trace(c);
    cerr << "Here 4" << endl;
    dd = trace(adj(a*b)*b*a*b);
    cerr << "Here 5" << endl;
    write(xml,"diff", Real(norm2(cc-dd)));
    dd = localInnerProduct(a,b);
    cerr << "Here 6" << endl;
    write(xml,"diff", Real(norm2(cc-dd)));
  }
#endif

#if 1
  {
    SpinMatrix S;
    random(S);

    LatticePropagator q;
    random(q);

    LatticePropagator di_quark;
    random(di_quark);

    SpinMatrix gamma_1 = zero;
    SpinMatrix gamma_5 = zero;
//    LatticeComplex ps_rho = trace(adj(gamma_5 * q * gamma_5) * gamma_1 * q * gamma_1);
//    LatticeComplex ps_rho = trace(adj(gamma_5 * q * gamma_5) * q * gamma_1);
    LatticeComplex ps_rho = trace(adj(gamma_5 * q) * gamma_1);
//    LatticeComplex ps_rho = localInnerProduct(gamma_5 * q, gamma_1);
//    Complex ps_rho = trace(adj(gamma_5 * q) * gamma_1);
//    LatticeComplex ps_rho = trace(adj(q) * gamma_1);
//    LatticeComplex ps_rho = localInnerProduct(q, gamma_1);

    cerr << "Here 1" << endl;
    LatticeComplex b = trace(S * traceColor(q * di_quark));

    cerr << "Here 2" << endl;
    di_quark = quarkContract13(q * Gamma(5), Gamma(5) * q);
    LatticeComplex c = trace(S * traceColor(q * traceSpin(di_quark)));

    cerr << "Here 3" << endl;
    LatticeComplex d = trace(S * traceColor(q * traceSpin(quarkContract13(q * Gamma(5), 
									  Gamma(5) * q))));

    cerr << "Here 4" << endl;
    write(xml,"diff", Real(norm2(c-d)));

    c = trace(q * di_quark);
    LatticeReal r = real(c);

    cerr << "Here 5" << endl;
    LatticeReal s = real(trace(q * di_quark));
    
    cerr << "Here 6" << endl;
    write(xml,"diff", Real(norm2(r-s)));

    cerr << "Here 7" << endl;
    s = real(trace(adj(q) * di_quark * q));

    cerr << "Here 8" << endl;
    s = real(trace(q * di_quark * q));
  }
#endif
  

  pop(xml);
  xml.close();

  // Time to bolt
  QDP_finalize();

  exit(0);
}
