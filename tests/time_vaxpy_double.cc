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


  tests.addTest(new time_VAXPY(), "time_AXPY" );
  tests.addTest(new time_VAXPYZ(), "time_AXPYZ" );
  tests.addTest(new time_QDP_PEQ(), "time_QDP_PEQ" );
  tests.addTest(new time_QDP_AXPYZ(), "time_QDP_AXPYZ" );
 
  tests.run();
  // Testjig is destroyed
  tests.summary();
}

