// $Id: t_remote.cc,v 1.3 2003-06-07 19:09:32 edwards Exp $

#include <iostream>
#include <fstream>

#include "qdp_filebuf.h"

using namespace QDPUtil;
using namespace std;


static const char* rtinode = "qcdi01.jlab.org";



int main(int argc, char *argv[])
{
  // initialize remote file service (QIO)
  RemoteFileInit(rtinode, true);

  {
#if 0
    // Remote output buffer and conventional stream
    RemoteOutputFileBuf ob;
    ob.open("test.out",std::ofstream::out | std::ofstream::trunc);
    std::ostream outfile(&ob);
#else
    // Remote output stream
    RemoteOutputFileStream outfile;
    outfile.open("test.out",std::ofstream::out | std::ofstream::trunc);
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
    ib.open("test.out",std::ifstream::in);
    std::istream infile(&ib);
#else
    // Remote input stream
    RemoteInputFileStream infile;
    infile.open("test.out",std::ifstream::in);
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
  RemoteFileShutdown();

  exit(0);
}
