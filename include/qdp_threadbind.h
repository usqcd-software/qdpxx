#ifndef QDPXX_THREADBIND_H
#define QDPXX_THREADBIND_H

// Thread binding interface for use on BlueGenes
namespace __QDP__
{
  void setThreadAffinity(int nCores, int threadsPerCore);
  void reportAffinity(void);
};




#endif
