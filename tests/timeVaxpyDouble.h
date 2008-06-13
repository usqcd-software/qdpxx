#ifndef TIME_VAXPY3
#define TIME_VAXPY3


#ifndef UNITTEST_H
#include "unittest.h"
#endif

class time_QDP_PEQ : public TestFixture { public: void run(void); };
class time_QDP_AXPYZ : public TestFixture { public: void run(void); };
class time_VAXPYZ : public TestFixture { public: void run(void); };
class time_VAXPY : public TestFixture { public: void run(void); };
#endif
