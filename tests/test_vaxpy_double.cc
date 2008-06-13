#include "qdp.h"
#include "unittest.h"
#include "testvol.h"

#include "testVaxpyDouble.h"

using namespace QDP;

int main(int argc, char **argv)
{
  // Initialize UnitTest jig
  TestRunner  tests(&argc, &argv, nrow_in);
  tests.addTest(new testVaxpy4_1(), "testVaxpy4_1" );
  tests.addTest(new testVaxpy4_2(), "testVaxpy4_2" );
  tests.addTest(new testVaxpy4_3(), "testVaxpy4_3" );

  tests.addTest(new testVaxpy4_RB0_1(), "testVaxpy4_RB0_1" );
  tests.addTest(new testVaxpy4_RB0_2(), "testVaxpy4_RB0_2" );
  tests.addTest(new testVaxpy4_RB0_3(), "testVaxpy4_RB0_3" );

  tests.addTest(new testVaxpy4_RB1_1(), "testVaxpy4_RB1_1" );
  tests.addTest(new testVaxpy4_RB1_2(), "testVaxpy4_RB1_2" );
  tests.addTest(new testVaxpy4_RB1_3(), "testVaxpy4_RB1_3" );

  tests.addTest(new testVaxpy4_RB30_1(), "testVaxpy4_RB30_1" );
  tests.addTest(new testVaxpy4_RB30_2(), "testVaxpy4_RB30_2" );
  tests.addTest(new testVaxpy4_RB30_3(), "testVaxpy4_RB30_3" );

  tests.addTest(new testVaxpy4_RB31_1(), "testVaxpy4_RB31_1" );
  tests.addTest(new testVaxpy4_RB31_2(), "testVaxpy4_RB31_2" );
  tests.addTest(new testVaxpy4_RB31_3(), "testVaxpy4_RB31_3" );

  tests.addTest(new testVaxpy4_RB0_PEQ_1(), "testVaxpy4_RB0_PEQ_1" );
  tests.addTest(new testVaxpy4_RB0_PEQ_2(), "testVaxpy4_RB0_PEQ_2" );
  tests.addTest(new testVaxpy4_RB0_PEQ_3(), "testVaxpy4_RB0_PEQ_3" );

  tests.addTest(new testVaxpy4_RB30_PEQ(), "testVaxpy4_RB30_PEQ" );

  // Run all tests
  tests.run();

  // Testjig is destroyed
  tests.summary();
}

