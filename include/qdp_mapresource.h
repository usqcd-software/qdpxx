#ifndef QDP_MAPRESOURCE
#define QDP_MAPRESOURCE

#include "qmp.h"

namespace QDP {

  // The MPI resources class for an FnMap.
  // An instance for each (dest/src node,msg_size) combination
  // exists so they can be reused over the whole program lifetime.
  // Can't allocate resources in constructor, since I use ::operator new
  // to allocate a whole array of them. This is necessary since if a 
  // size of 2 occurs in the logical machine grid, then forward/backward
  // points to the same MPI node.

struct FnMapRsrc
{
  FnMapRsrc(const FnMapRsrc&) = delete;
  //FnMapRsrc() = delete;
  FnMapRsrc():bSet(false) {};

  void setup(int _destNode,int _srcNode,int _sendMsgSize,int _rcvMsgSize);
  void cleanup();

  ~FnMapRsrc() {
    //QDPIO::cout << "~FnMapRsrc()\n";
  }

  void qmp_wait() const;
  void send_receive() const;

  template<typename T> const T* getSendBufPtr() const { return static_cast<T*>(send_buf); }
  template<typename T> const T* getRecvBufPtr() const { return static_cast<T*>(recv_buf); }

  bool bSet;
  mutable void * send_buf;
  mutable void * recv_buf;
  int srcnum, dstnum;
  QMP_msgmem_t msg[2];
  QMP_msghandle_t mh_a[2], mh;
  QMP_mem_t *send_buf_mem;
  QMP_mem_t *recv_buf_mem;
};

  // This wrapper is used, since FnMap's are copy-constructed
  // during recursing down the expression with forEach. Thus a
  // shared_ptr to this wrapper ensures that the original and copy
  // point to the same resource class.

class RsrcWrapper
{
  const FnMapRsrc* pRsrc;
  bool rAlloc;
public:
  RsrcWrapper(): rAlloc(false) {}
  const FnMapRsrc* get() const {
    if (!rAlloc)
      QDP_error_exit("RsrcWrapper::get() internal error");
    return pRsrc;
  }
  void set(const FnMapRsrc* p) { rAlloc=true; pRsrc=p;}
};


  // A 2D container of resource classes.
  // We index msg_size and dest_node and take the
  // index as the coordinate in the 2D matrix.

class FnMapRsrcMatrix {

  multi2d<FnMapRsrc*> m2d;
  std::vector<int> sendMsgSize;
  std::vector<int> destNode;
  int numSendMsgSize;
  int numDestNode;
  int maxDeep;

  FnMapRsrcMatrix(): maxDeep(32),
		     numSendMsgSize(32), 
		     numDestNode(Nd*2), 
		     sendMsgSize(0), 
		     destNode(0) {
    m2d.resize(numSendMsgSize,numDestNode);

    for(int i=0;i<numSendMsgSize;i++) {
      for(int q=0;q<numDestNode;q++) {
	m2d(i,q)=NULL;
      }
    }
  }


  public:

  void cleanup() {
    QDPIO::cout << "FnMapRsrcMatrix cleanup\n";

    for(int i=0;i<numSendMsgSize;i++) {
      for(int q=0;q<numDestNode;q++) {
	//QDPIO::cout << "m2d" << i << " " << q << " = " << m2d(i,q) << "\n";
	//printf("%p\n",m2d(i,q));
	if (m2d(i,q)) {
	  for (int e = 0 ; e < maxDeep ; e++ ) {
	    //QDPIO::cout << "cleanup" << i << " " << q << " " << e << "\n";
	    m2d(i,q)[e].cleanup();
	  }
	  delete[] m2d(i,q);
	}
      }
    }
  }


  const FnMapRsrc& get(int _destNode,int _srcNode,int _sendMsgSize,int _rcvMsgSize, int shift_num) {
    bool found = false;
    int xDestNode=0;
    for(; xDestNode < destNode.size(); ++xDestNode)
      if (destNode[xDestNode] == _destNode)
	{
	  found = true;
	  break;
	}
    if (! found) {
      if (destNode.size() == numDestNode) {
	QDP_error_exit("FnMapRsrcMatrix not enough space in destNode");
      } else {
	destNode.push_back(_destNode);
	xDestNode=destNode.size()-1;
      }
    }
    //QDPIO::cout << "using node place = " << xDestNode << "\n";


    found = false;
    int xSendmsgsize=0;
    for(; xSendmsgsize < sendMsgSize.size(); ++xSendmsgsize)
      if (sendMsgSize[xSendmsgsize] == _sendMsgSize)
	{
	  found = true;
	  break;
	}
    if (! found) {
      if (sendMsgSize.size() == numSendMsgSize) {
	QDP_error_exit("FnMapRsrcMatrix not enough space in sendmsgsize");
      } else {
	sendMsgSize.push_back(_sendMsgSize);
	xSendmsgsize=sendMsgSize.size()-1;
      }
    }
    //QDPIO::cout << "using msg_size place = " << xSendmsgsize << "\n";

    if (!m2d(xSendmsgsize,xDestNode)) {
      QDPIO::cout << "setup " << maxDeep << " map resource objs:";
      m2d(xSendmsgsize,xDestNode) = new FnMapRsrc[maxDeep];
      for (int i = 0 ; i < maxDeep ; i++ ) {
	QDPIO::cout << ".";
	m2d(xSendmsgsize,xDestNode)[i].setup( _destNode, _srcNode, _sendMsgSize, _rcvMsgSize );
      }
      QDPIO::cout << "\n";
    }
    //QDPIO::cout << "returning shift_num = " << shift_num << "\n";

    //printf("0,0=%p 1,0=%p 2,0=%p %dx%d\n",m2d(0,0),m2d(1,0),m2d(2,0),m2d.size1(),m2d.size2());

    return m2d(xSendmsgsize,xDestNode)[shift_num];
  }

  static FnMapRsrcMatrix& Instance() {
    static FnMapRsrcMatrix singleton;
    return singleton;
  }

};

} // namespace QDP

#endif
