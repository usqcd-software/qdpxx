// $Id: t_remote.cc,v 1.2 2003-06-06 15:10:46 edwards Exp $

#include <iostream>
#include <fstream>

#include "qdp_filebuf.h"

using namespace QDPUtil;
using namespace std;


extern "C"
{
  int qio_init(const char *node, int rtiP0);
  void qio_shutdown();
}

static const char* rtinode = "qcdi01.jlab.org";



int main(int argc, char *argv[])
{
  // initialize remote file service (QIO)
  qio_init(rtinode, 1);

  {
#if 0
    // Remote output buffer and conventional stream
    RemoteOutputFileBuf ob;
    ob.open("/home/edwards/test.out",std::ofstream::out | std::ofstream::trunc);
    std::ostream outfile(&ob);
#else
    // Remote output stream
    RemoteOutputFileStream outfile;
    outfile.open("/home/edwards/test.out",std::ofstream::out | std::ofstream::trunc);
#endif

    int n=42;
    float x = -5.3;

    outfile << n << endl;
    outfile << x << endl;

    outfile.close();
  }
 
  {
#if 0
    // Remote input buffer and conventional stream
    RemoteInputFileBuf ib;
    ib.open("/home/edwards/test.out",std::ifstream::in);
    std::istream infile(&ib);
#else
    // Remote input stream
    RemoteInputFileStream infile;
    infile.open("/home/edwards/test.out",std::ifstream::in);
#endif

    int nn;
    float xx;

    infile >> nn;
    infile >> xx;

    cerr << "nn = " << nn << endl;
    cerr << "xx = " << xx << endl;

    infile.close();
  }

  // shutdown remote file service (QIO)
  qio_shutdown();

  return 0;
}
