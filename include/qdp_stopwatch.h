// -*- C++ -*-
// $Id: qdp_stopwatch.h,v 1.4 2005-10-17 04:27:36 edwards Exp $
/*! @file
 * @brief Timer support
 *
 * A stopwatch like timer.
 */

#include<sys/time.h>

#ifndef QDP_STOPWATCH_H
#define QDP_STOPWATCH_H

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup timer Timer
 *
 * @ingroup qdp
 *
 * @{
 */
class StopWatch 
{
public:
  //! Constructor
  StopWatch();

  //! Destructor
  ~StopWatch();

  //! Reset the timer
  void reset();

  //! Start the timer
  void start();

  //! Stop the timer
  void stop();

  //! Get time in microseconds
  double getTimeInMicroseconds();

  //! Get time in seconds
  double getTimeInSeconds();

private:
  long sec;
  long usec;
  bool startedP;
  bool stoppedP;

  struct timeval t_start;
  struct timeval t_end;
};

QDP_END_NAMESPACE();
  
#endif
