#ifndef QDP_STOPWATCH_H
#define QDP_STOPWATCH_H

QDP_BEGIN_NAMESPACE(QDP);

#include<sys/time.h>

class StopWatch { 
 public:
  StopWatch(void) {
    stoppedP=false;
    startedP=false;

  }

  ~StopWatch(void) {};

  void reset(void) {
    startedP = false;
    stoppedP = false;
  }

  void start(void) {
    int ret_val;
    ret_val = gettimeofday(&t_start, NULL);
    if( ret_val != 0 ) { 
      QDP_error_exit("Gettimeofday failed in StopWatch::start()\n");
    }
    startedP = true;
    stoppedP = false;
  }

  void stop(void) {
    if( !startedP ) { 
      QDP_error_exit("Attempting to stop a non running stopwatch in StopWatch::stop()\n");
    }

    int ret_val;
    ret_val = gettimeofday(&t_end, NULL);
    if( ret_val != 0 ) {
      QDP_error_exit("Gettimeofday failed in StopWatch::end()\n");
    }
    stoppedP = true;
  }

  double getTimeInMicroseconds(void) {
    long usecs;
    if( startedP && stoppedP ) { 
      if( t_end.tv_sec < t_start.tv_sec ) { 
	QDP_error_exit("Critical timer rollover\n");
      }
      else { 
	usecs = (t_end.tv_sec - t_start.tv_sec)*1000000;


	if( t_end.tv_usec < t_start.tv_usec ) {
	  usecs -= 1000000;
	  usecs += 1000000+t_end.tv_usec - t_start.tv_usec;
	}
	else {
	  usecs += t_end.tv_usec - t_start.tv_usec;
	}
      }
    }
    else {
      QDP_error_exit("Either stopwatch not started, or not stopped\n");
    }

    return (double)usecs;
  }
    
  double getTimeInSeconds(void)  {
    double t_sec = getTimeInMicroseconds() / 1e6;   
    return t_sec;
  }

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
