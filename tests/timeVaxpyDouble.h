#ifndef TIME_VAXPY3
#define TIME_VAXPY3


#ifndef UNITTEST_H
#include "unittest.h"
#endif

class time_QDP_PEQ : public TestFixture { public: void run(void); };
class time_QDP_AXPYZ : public TestFixture { public: void run(void); };
class time_VAXPYZ : public TestFixture { public: void run(void); };
class time_VAXPY : public TestFixture { public: void run(void); };
class time_VAXMY : public TestFixture { public: void run(void); };
class time_VAXMYZ : public TestFixture { public: void run(void); };
class time_VAXPBYZ : public TestFixture { public: void run(void); };
class time_VAXPBY : public TestFixture { public: void run(void); };
class time_VSCAL : public TestFixture { public: void run(void); };

class time_LOCAL_SUMSQ : public TestFixture { public: void run(void); };
class time_LOCAL_VCDOT : public TestFixture { public: void run(void); };
class time_LOCAL_VCDOT_REAL : public TestFixture { public: void run(void); };

class time_SUMSQ : public TestFixture { public: void run(void); };
class time_VCDOT : public TestFixture { public: void run(void); };
class time_VCDOT_REAL : public TestFixture { public: void run(void); };

#endif
