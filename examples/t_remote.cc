// $Id: t_remote.cc,v 1.1 2003-06-06 02:39:30 edwards Exp $

#include <iostream>

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
    ob.open("test.out");
    std::ostream outfile(&ob);
#else
    // Remote output stream
    RemoteOutputFileStream outfile;
    outfile.open("test.out");
#endif

    int n=42;
    float x = -5.3;

    outfile << "n = " << n << endl;
    outfile << "x = " << x << endl;

    outfile.close();
  }
 

  {
#if 0
    // Remote input buffer and conventional stream
    RemoteInputFileBuf ib;
    ib.open("test.out");
    std::istream infile(&ib);
#else
    // Remote input stream
    RemoteInputFileStream infile;
    infile.open("test.out");
#endif

    int n;
    float x;

    infile >> n;
    infile >> x;

    cerr << "n = " << n << endl;
    cerr << "x = " << x << endl;

    infile.close();
  }

  // shutdown remote file service (QIO)
  qio_shutdown();

  return 0;
}
