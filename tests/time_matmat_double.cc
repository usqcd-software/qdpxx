#include "qdp.h"
#include "unittest.h"
#include "testvol.h"


#include "timeMatEqMatMatDouble.h"

using namespace QDP;

int main(int argc, char **argv)
{
  // Initialize UnitTest jig
  TestRunner  tests(&argc, &argv, nrow_in);
  QDPIO::cout << "Volume= { " << Layout::lattSize()[0]
	      << " , " << Layout::lattSize()[1]
	      << " , " << Layout::lattSize()[2]
	      << " , " << Layout::lattSize()[3] << " } " << endl;

  
  //  tests.addTest(new timeMeqMM_QDP(), "timeMeqMM_QDP" );
  // tests.addTest(new timeMeqMM(), "timeMeqMM" );

  tests.addTest(new timeMPeqaMM_QDP(), "timeMPeqaMM_QDP" );
  tests.addTest(new timeMPeqaMM(), "timeMPeqaMM" );
  // Run all tests
  tests.run();

  // Testjig is destroyed
  tests.summary();
}

