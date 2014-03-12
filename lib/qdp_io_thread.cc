/*
 * Copyright (C) <2009> Jefferson Science Associates, LLC
 *                      Under U.S. DOE Contract No. DE-AC05-06OR23177
 *
 *                      Thomas Jefferson National Accelerator Facility
 *
 *                      Jefferson Lab
 *                      Scientific Computing Group,
 *                      12000 Jefferson Ave.,      
 *                      Newport News, VA 23606 
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * ----------------------------------------------------------------------------
 * Description:
 *     QDP++ I/O Thread and related classes
 *
 *
 * Author:
 *     Jie Chen
 *     Scientific Computing Group
 *     Jefferson Lab
 *
 */
#include "qdp_io_thread.h"
#include <signal.h>
#include <sys/time.h>
#include <stdlib.h>

namespace QDPIOThread
{
  //========================================================================
  // Implementation of ioCond Class
  //========================================================================
  ioCond::ioCond (ioMutex& lock)
    : mutex_ (lock)
  {
    if (pthread_cond_init (&cond_, 0) != 0) {
      fprintf (stderr, "cannot initialize pthread conditional variable. \n");
      ::exit (1);
    }
  }

  ioCond::~ioCond (void)
  {
    pthread_cond_destroy(&cond_);
  }


  int
  ioCond::wait (float timeout)
  {
    int status = 0;

    if (timeout >= 0.0000001) {
      struct timespec cts;
      struct timespec ts;

      struct timeval tts;
      gettimeofday (&tts, 0);

      cts.tv_sec = tts.tv_sec;
      cts.tv_nsec = 1000*tts.tv_usec;

      // get interval time
      ts.tv_sec = (time_t)timeout;
      ts.tv_nsec = (timeout - (time_t)timeout)*1000000000;

      // get absolut time
      cts.tv_sec += ts.tv_sec;
      cts.tv_nsec += ts.tv_nsec;

      int runstatus = 0;
      if ((runstatus = pthread_cond_timedwait (&cond_, &mutex_.lock_, &cts))
	  == ETIME)
	status = 1;
      else if (runstatus == -1)
	status = -1;
    }
    else
      wait ();

    return status;
  }

  void
  ioCond::wait (void)
  {
    if (pthread_cond_wait (&cond_, &mutex_.lock_) < 0) {
      fprintf (stderr, "Cannot wait on a pthread conditional variable. \n");
      ::exit (1);
    }
  }

  void
  ioCond::signal (void)
  {
    if (pthread_cond_signal(&cond_) != 0) {
      fprintf (stderr, "Cannot signal a pthread conditional variable. \n");
      ::exit (1);
    }
  }

  void
  ioCond::broadcast (void)
  {
    if (pthread_cond_broadcast (&cond_) != 0) {
      fprintf (stderr, "Cannot broadcast on a pthread conditional variable. \n");
      ::exit (1);
    }
  }


  //========================================================================
  // Implementation ioMonitor
  //========================================================================
  ioMonitor::ioMonitor (void)
    :mutex_(), mcond_ (mutex_)
  {
    // empty
  }

  ioMonitor::~ioMonitor(void)
  {
    // empty
  }

  void
  ioMonitor::wait (void)
  {
    mcond_.wait ();
  }

  int
  ioMonitor::wait (float timeout)
  {
    return mcond_.wait (timeout);
  }

  void
  ioMonitor::notify (void)
  {
    mcond_.signal ();
  }

  void
  ioMonitor::notifyAll (void)
  {

    mcond_.broadcast ();
  }

  int
  ioMonitor::lock (void)
  {
    return mutex_.lock ();
  }

  int
  ioMonitor::acquire (void)
  {
    return mutex_.lock ();
  }

  int
  ioMonitor::unlock (void)
  {
    return mutex_.unlock ();
  }

  int
  ioMonitor::release (void)
  {
    return mutex_.unlock ();
  }

  //===========================================================================
  //    Implementation of ioThread Class
  //===========================================================================

#define _IO_THREAD_NAME_PREFIX "Thread-"

  void *
  ioThread::startThread (void *arg)
  {
    ioThread* obj = (ioThread *)arg;
    
    obj->monitor_.lock();
    obj->threadRunning_ = 1;
    obj->monitor_.unlock ();
    obj->monitor_.notify();

    obj->run ();

    // object is created on heap
    // delete obj;

    return 0;
  }

  int
  ioThread::initThread (ioRunnable* target, const char* name)
  {
    target_ = target;
    id_ = 0;
  
    // create a new thread
    if (createThread () != 0)
      return -1;

    // must be called after create thread
    setName (name);

    return 0;
  }

  void
  ioThread::setName (const std::string& name)
  {
    char buffer[1024];

    if (!name.empty ())
      name_ = name;
    else {
      snprintf (buffer, sizeof(buffer), "%s%ld", _IO_THREAD_NAME_PREFIX, id_);
      name_ = std::string(buffer);
    }
  }

  int
  ioThread::createThread (void)
  {
    // get thread id pointer
    pthread_t tid;

    if(pthread_create (&tid, NULL, &(ioThread::startThread), (void *)this) != 0)
      return -1;
  
    // wait for thread is up before going further
    monitor_.lock ();
    while (!threadRunning_) {
      monitor_.wait ();
    }
    id_ = tid;

    monitor_.unlock ();
    return 0;
  }

  ioThread *
  ioThread::create (ioRunnable* target,
		    const char* name)
  {
    ioThread* nt = new ioThread (target, name);
    return nt;
  }
  
  ioThread::ioThread (ioRunnable* target,const char* name)
    :name_(), target_(0), threadRunning_(0), monitor_()
  {
    if (initThread(target, name) == 0)
      target->thread (this);
  }

  ioThread::~ioThread (void)
  {
    // empty
    // target delete this thread
  }

  const std::string&
  ioThread::getName (void) const
  {
    return name_;
  }

  int
  ioThread::kill (int signum)
  {
    return pthread_kill (id_, signum);
  }

  int
  ioThread::join (void)
  {
    void* retstatus;
    return pthread_join (id_, &retstatus);
  }

  void
  ioThread::run (void)
  {
    if (target_)
      target_->run ();
  }

  int
  ioThread::sleep (long ms, long ns)
  {
    struct timespec tv;
    tv.tv_sec = ms/1000;
    tv.tv_nsec = (ms% 1000)*1000000 + ns;

    if (nanosleep (&tv, 0) < 0 && errno == EINTR)
      return 1; 

    return 0;
  }

  void
  ioThread::yield (void)
  {
    pthread_yield();
  }

  pthread_t
  ioThread::getID (void) const
  {
    return id_;
  }

  unsigned long
  ioThread::currentTime (void)
  {
    unsigned long t;
    struct timeval tv;
    gettimeofday(&tv, 0);
    t = tv.tv_sec * 1000;
    t += tv.tv_usec/1000;

    return t;
  }
}
