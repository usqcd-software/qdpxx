#include "unittest.h"
#include "testvol.h"

#include <string>
#include "timeVaxpyDouble.h"

using namespace QDP;
using namespace std;

int main(int argc, char **argv)
{
  // Initialize UnitTest jig
  TestRunner  tests(&argc, &argv, nrow_in);
  QDPIO::cout << "Volume= { " << Layout::lattSize()[0]
	      << " , " << Layout::lattSize()[1]
	      << " , " << Layout::lattSize()[2]
	      << " , " << Layout::lattSize()[3] << " } " << endl;


  // This behaves as expected
  //  tests.addTest(new time_VAXPBYZ(), "time_AXPBYZ" );
  // tests.addTest(new time_VAXPBY(), "time_AXPBY" );  
  // tests.addTest(new time_VAXMBYZ(), "time_AXMBYZ" );
  // tests.addTest(new time_VAXMBY(), "time_AXMBY" );  
  tests.addTest(new time_VAXPYZ(), "time_AXPYZ" );
  tests.addTest(new time_VAXPY(), "time_AXPY" );

  // 
  // tests.addTest(new time_VAXMY(), "time_AXMY" );
  //  tests.addTest(new time_VAYPX(), "time_AYPX" );

  //tests.addTest(new time_LOCAL_SUMSQ(), "time_LOCAL_SUMSQ");
  // tests.addTest(new time_LOCAL_VCDOT(), "time_LOCAL_VCDOT");
  // tests.addTest(new time_LOCAL_VCDOT_REAL(), "time_LOCAL_VCDOT_REAL");


#if 0



  tests.addTest(new time_VAXMYZ(), "time_AXMYZ" );

  tests.addTest(new time_VSCAL(), "time_VSCAL");


  tests.addTest(new time_SUMSQ(), "time_SUMSQ");
  tests.addTest(new time_VCDOT(), "time_VCDOT");
  tests.addTest(new time_VCDOT_REAL(), "time_VCDOT_REAL");

#endif
  // tests.addTest(new time_QDP_PEQ(), "time_QDP_PEQ" );
  // tests.addTest(new time_QDP_AXPYZ(), "time_QDP_AXPYZ" );

 
  tests.run();
  // Testjig is destroyed
  tests.summary();
}

