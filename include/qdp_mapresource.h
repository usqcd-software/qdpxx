#ifndef QDP_MAPRESOURCE
#define QDP_MAPRESOURCE

#include "qmp.h"

namespace QDP {

struct FnMapRsrc
{
  FnMapRsrc(const FnMapRsrc&) = delete;
  //FnMapRsrc() = delete;
  FnMapRsrc():bSet(false) {};

  void setup(int _destNode,int _srcNode,int _sendMsgSize,int _rcvMsgSize);
  void cleanup();

  //~FnMapRsrc();

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
