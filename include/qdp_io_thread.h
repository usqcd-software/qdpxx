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
#pragma once

#include <cstdio>
#include <string>
#include <errno.h>

namespace QDPIOThread
{
  class ioCond;

  class ioMutex 
  {
    pthread_mutex_t  lock_;

  public:
    /**
     *   Constructor for ioMutex 
     */
    ioMutex (void) {
      ::pthread_mutex_init(&lock_, 0);
    }

    /**
     *   Destructor for ioMutex
     */
    virtual ~ioMutex (void) {
      ::pthread_mutex_destroy (&lock_);
    }

    /**
     * Lock operation
     *  
     *      Be careful when you call lock if you already have it.
     *      return 0 on success
     */
    int lock (void) {
      return pthread_mutex_lock (&lock_);
    }

    /**
     * Diffrent name for lock
     */
    int acquire (void)
    {
      return this->lock ();
    }


    /**
     * unlock this mutex
     *      return 0 on success
     */
    int unlock (void)
    {
      return pthread_mutex_unlock (&lock_);
    }

    /**
     * Different name for unlock
     */
    int release (void)
    {
      return this->unlock ();
    }

    /**
     * Non block lock.
     *   return 0 if one gets it.
     *   return 1 if someone else has the lock.
     *   return -1 if an internal error occurs.
     */
    int tryLock   (void)
    {
      if (::pthread_mutex_trylock (&lock_) != 0) {
	if (errno == EBUSY)
	  return 1;
	else {
	  return -1;
	}
      }
      return 0;
    }

    /**
     * A different name for tryLock.
     */
    int tryAcquire (void) {
      return this->tryLock ();
    }
    
  private:
    // deny assignment and copy operations
    ioMutex (const ioMutex& mutex);
    ioMutex& operator = (const ioMutex& mutex);

    friend class ioCond;
  };

  /*************************************************************************
   *  Condition class: it is used to signal other threads waiting on       *
   *   a condition                                                         *
   *************************************************************************/
  class ioCond
  {
  public:
    /**
     * Constructor using a mutex
     */
    ioCond (ioMutex& lock);
  
    /**
     * Destrcutor
     */
    virtual ~ioCond (void);

  
    /**
     * wait for conditional variable to be signaled
     * return 0 on success.
     * return 1 timedout.
     * return others: bad
     *
     * timeout > 0.0000001 considers a timed wait.
     */
    int  wait (float timeout);

    /**
     * Wait forever for a conditional variable.
     */
    void wait (void);

   
    /**
     * signal one waiting thread.
     */
    void signal (void);

    /** 
     * signal all waiting thread.
     */
    void broadcast (void);
    
  private:
    pthread_cond_t cond_;

    ioMutex&       mutex_;

    // hide copy and assignment operators
    ioCond (const ioCond& cond);
    ioCond& operator = (const ioCond& cond);
  };

  /*************************************************************************
   * Monitor Class: This class can be used to ensure there is only a       *
   * single thread accessing internals of an object at a time.             *
   * Application classes that are derived from this class can use          *
   * ioSynchronized to do automatic scope locking.                         *
 ************************************************************************/
  class ioSynchronized;

  class ioMonitor
  {
  public:
    /**
     * Constructor
     */
    ioMonitor (void);

    /**
     * destructor
     */
    virtual ~ioMonitor (void);

    /**
     * wait for notification
     */
    virtual void  wait (void);

    /**
     * wait for notification
     * return 0: normal return
     * return -1: error, return IO_TIMEOUT: timeout
     */
    virtual int   wait (float time);

    /**
     * signal one object
     */
    void    notify (void);

    /**
     * signal all
     */
    void    notifyAll (void);

    /**
     * Lock this class scope
     */
    int     lock    (void);
    int     acquire (void);

    /** 
     * Unlock this class scope.
     */
    int     unlock  (void);
    int     release (void);

  protected:

    friend class ioSynchronized;

    ioMutex          mutex_;
    ioCond           mcond_;
  };


  /*************************************************************************
   * Synchronized Class: This class can be used along with ioMutex         *
   * or ioMonitor to do automatic lock and unlock a critical section       *
   *************************************************************************/
  class ioSynchronized
  {
  private:
    // hide copy and assignment operator
    ioSynchronized (const ioSynchronized& sync);
    ioSynchronized& operator = (const ioSynchronized& sync);
  
    ioMutex&   lock_;

  public:
    
    /**
     * Constructor using ioMutex
     */
    ioSynchronized (ioMutex& mutex);

    
    /**
     * Destructor.
     */
    ~ioSynchronized (void) {
      lock_.unlock ();
    }
  };

  inline
  ioSynchronized::ioSynchronized(ioMutex& mutex)
    :lock_ (mutex)
  {
    lock_.lock ();
  }

  typedef ioSynchronized ioGuard;

#define QDP_GUARD(s) ioGuard guard(s)

  
  /**
   * Forward decleration of ioThread
   */
  class ioThread;

  /**
   * Runnable class
   * This class is an abstract class. Any class want to be a seperate thread
   * need to inherit this class
   *
   * This class can be used only by calling new
   */
  class ioRunnable
  {
  private:
    ioThread*             thread_;

    friend class          ioThread;

  protected:
    /**
     * Constructor: no direct instantiation
     */
    ioRunnable (void) 
      :thread_(0)
    {
      // empty
    }

    /**
     * Destructor
     */
    virtual ~ioRunnable (void)
    {
      // empty
    }

    /**
     * Set up thread pointer
     */
    void  thread (ioThread* ptr) {thread_ = ptr;}

  public:

    /**
     * Subclasses must implement this function.
     */
    virtual void run (void) = 0;


    /**
     * Return a thread pointer
     */
    ioThread* thread (void) const {return thread_;}
  };

  /**
   * Thread class. This is a private class which should not be used directly
   */
  class ioThread
  {
  public:
    /**
     * Creation methods for thread
     *    ioThread can not be created on stack.
     */
    static ioThread* create (ioRunnable* target, const char* name = 0);


    /**
     * get name of this thread.
     */
    const std::string& getName (void) const;

    /**
     * wait for this thread to finish.
     */
    int         join (void);

    /**
     * if this thread was constructed using a separate Runnable object,
     * that obj->run method called, otherwise this does nothing.
     */
    virtual void run (void);

    /**
     * set name of this thread.
     */
    void    setName   (const std::string& name);

    /**
     * cause current thread to sleep for millisecond long.
     */
    int     sleep     (long ms, long ns = 0);

    /**
     * send a signal to thread.
     */
    int     kill      (int signum);

    /** 
     * cause this thread to give up CPU.
     */
    void    yield     (void);

    /**
     * return threadId of this thread.
     */
    pthread_t getID (void) const;

    // delete this thread explictly from target
    virtual ~ioThread (void);

    /**
     * return current time in milli second
     */
    static unsigned long currentTime (void);

  protected:
   
    /**
     * Constrctor from a runnable pointer
     */
    ioThread (ioRunnable* target, const char* name = 0);


    // this is thread function called by pthread_create
    static void* startThread (void *arg);

    // init thread method called by constructors
    int  initThread (ioRunnable* target, const char* name);

    // create a thread called by the above
    int  createThread (void);

    // exit part of this thread
    void     threadExit (void);


  private:
    // data area
    std::string name_; // Name of this thread
    ioRunnable* target_;  // really the running object
    // id for this thread
    pthread_t id_;
    // flag to tell thread is running
    int threadRunning_;
    // monitor for new thread
    ioMonitor monitor_;
  };
}
