#ifndef UNITTEST_H
#define UNITTEST_H

#include "qdp.h"
#include <vector>

using namespace QDP;

namespace Assertions { 
  template<typename T>
  inline 
  void assertEquals(const T& t1, const T& t2) {
    if ( t1 != t2 ) { 
      throw std::exception();
    }
  }

  template<typename T>
  inline
  void assertNotEquals(const T& t1, const T& t2) { 
    if( t1 == t2 ) { 
      throw std::exception();
    }
  }

  // Add other assertions here
};

// A test case - provides a run() method and a getName() to identify itself
// Strictly speaking the getName() is probably not necessary
class TestCase {
private:
public:
  virtual void run(void) = 0;
  virtual const std::string getName(void) const = 0;
};

// A test fixture - that does extra set up before and after the test
class TestFixture : public TestCase {
public:
  virtual void setUp() {} ;
  virtual void tearDown() {};
  virtual void runTest() {};
  virtual const std::string getName() const = 0; 
  
  void run(void) { 
      setUp();
      runTest();
      tearDown();
  }
};

// A runner class, which holds a bunch of tests
// Essentially this is a testCase,  but it is special
// As it sets up QDP as well -- can probably fudge things 
// to get the lattice dimensions from argc, and argv 
class TestRunner : public TestCase { 
private:
  int num_success;
  int num_failed;
  int num_unexpected_failed;
  int num_tried;

  std::vector<TestCase*> tests;
  

 public: 
  TestRunner(int* argc, char ***argv, const int latdims[]) : 
    num_success(0),
    num_failed(0),
    num_unexpected_failed(0),
    num_tried(0)
  {
    QDP_initialize(argc, argv);
    multi1d<int> nrow(Nd);
    nrow = latdims;
    Layout::setLattSize(nrow);
    Layout::create();
  }

  const std::string getName() const { return string("TestRunner"); }

  void run(void) {    
     for( int i=0; i != tests.size(); i++) {
      // Run ith test. We have a vector of pointer so must dereference tests[i]
      run(*(tests[i]));
    }
  }  

  
  // Extra Features: Add a test
  void addTest(TestCase* t) { 
    if( t != 0x0 ) { 
      tests.push_back(t);
    }
  }

  // Run a particular test
  void run(TestCase& t) { 

    try {
      QDPIO::cout << "Running Test: " << t.getName();
      num_tried++;
      t.run();
      QDPIO::cout << " OK" << endl;
      num_success++;
    }
    catch( std::exception ) { 
      QDPIO::cout << " FAILED" << endl;
      num_failed++;
    }
    catch(...) { 
      QDPIO::cout << " UNEXPECTED FAILURE" << endl;
      num_failed++;
      num_unexpected_failed++;

    } 
  }


  void summary() { 
    QDPIO::cout << "Summary: " << num_tried <<   " Tests Tried" << endl;
    QDPIO::cout << "         " << num_success << " Tests Succeeded " << endl;
    QDPIO::cout << "         " << num_failed  << " Tests Failed " << endl;
    QDPIO::cout << "of which " << num_unexpected_failed << " Tests Failed in Unexpected Ways" << endl;
  }


  ~TestRunner() { 
    
    for( std::vector<TestCase*>::iterator i=tests.begin(); 
	 i != tests.end(); i++) {
      delete(*i);
    }
    QDP_finalize();
  }

 private:
 

};

#endif
