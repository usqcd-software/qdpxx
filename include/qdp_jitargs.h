#ifndef QDP_JIT_ARGS
#define QDP_JIT_ARGS

#include<vector>

namespace QDP {

  class QDPJitArgs {
  public:
    enum UnionType { Ptr=0, Int=1, Bool=2, IntPtr=3, Size_t=4 };

    QDPJitArgs();
    ~QDPJitArgs();
    string getPtrName() const;
    string getCode(int i) const;
    UnionDevPtr* getDevPtr();
    int addPtr(void * devicePtr) const;
    int addInt(int i) const;
    int addBool(bool b) const;
    int addIntPtr( int * intPtr) const;
    int addSize_t(size_t i) const;

    friend bool operator== (const QDPJitArgs& a, const QDPJitArgs& b);

  private:
    mutable std::vector< UnionDevPtr > vecArgs;
    mutable std::vector< int >         vecType;
    static  std::vector< std::string > vecUnionMember;
    mutable int myId;
  };

  bool operator== (const QDPJitArgs& a, const QDPJitArgs& b);
  bool operator!= (const QDPJitArgs& a, const QDPJitArgs& b);

}

#endif
