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
 *     QDP++ I/O Writer Thread and related routines
 *     
 *
 * Author:
 *     Jie Chen
 *     Scientific Computing Group
 *     Jefferson Lab
 *
 */
#include "qdp_io_writer.h"
#include <stdlib.h>
#include <byteswap.h>

using namespace std;

namespace QDPIOThread
{ 
  //=================================================================
  // Implementation of WriteBuffer Class
  //=================================================================

  /**
   * constructor
   */
  WriteBuffer::WriteBuffer (unsigned int size)
    :bufsize_(size), datasize_(0), offset_(-1), 
     data_(0), type_ (WriteBuffer::REGULAR), 
     lock_()
  {
    // allocate space aligned on unsigned long
    // C++ makes sure tmp is aligned on unsigned long
    unsigned long *tmp = new unsigned long[size/sizeof(unsigned long) + 1];
    data_ = (unsigned char *)tmp;
  }

  /**
   * Destructor
   */
  WriteBuffer::~WriteBuffer (void)
  {
    delete []data_;
  }

  /**
   * Return total size of buffer
   */
  unsigned int 
  WriteBuffer::size (void) const
  {
    return bufsize_;
  }


  /**
   * Return data pointer
   */
  unsigned char* 
  WriteBuffer::data (void) const
  {
    return data_;
  }

  /**
   * Reset buffer to initial state
   */
  void
  WriteBuffer::reset (void)
  {
    QDP_GUARD(lock_);
    datasize_ = 0;
    type_ = WriteBuffer::REGULAR;
    offset_ = -1;
  }

    
  /**
   * Return current size of data
   */
  unsigned int 
  WriteBuffer::dataSize (void)
  {
    QDP_GUARD(lock_);
    return datasize_;
  }

  /**
   * Return current space for data
   */
  unsigned int 
  WriteBuffer::availableSpace (void)
  {
    QDP_GUARD(lock_);
    return bufsize_ - datasize_;
  }

  /**
   * Copy data into this buffer with option of swapping bytes
   */
  void
  WriteBuffer::copy (unsigned char* val, size_t size, size_t nelem, 
		     int swapbytes)
  {
    QDP_GUARD(lock_);

    if (swapbytes) {
      switch (size) {
      case 4:  // float or integer
	{
	  unsigned int *w = (unsigned int *)val;
	  unsigned int *d = (unsigned int *)&(data_[datasize_]);

	  for(int j = 0; j < nelem; j++) 
	    d[j] = bswap_32(w[j]);	  
	}
	break;
      case 8:  /* n_uint64_t or double */
	{
	  unsigned long *w = (unsigned long *)val;
	  unsigned long *d = (unsigned long *)&(data_[datasize_]);
	  for(int j = 0; j < nelem; j++)
	    d[j] = bswap_64(w[j]);	  
	}
	break;
      case 2:  /* n_uint16_t */
	{
	  unsigned short *w = (unsigned short *)val;
	  unsigned short *d = (unsigned short *)&(data_[datasize_]);
	  for(int j = 0; j < nelem; j++)
	    d[j] = bswap_16(w[j]);
	}
	break;
      case 1:  /* n_uint8_t: byte - do nothing */
	memcpy (&(data_[datasize_]), val, size * nelem);
	break;
      case 16:  /* Long Long */
	{
	  fprintf (stderr, "Not implemented for long long swap.\n");
	  exit (1);
	}
      default:
	std::cerr << __func__ << ": unsupported word size = " << size << "\n";
	exit(1);
      }
    }
    else
      memcpy (&(data_[datasize_]), val, size * nelem);

    datasize_ += (size * nelem);
  }

  /**
   * check this buffer type
   */
  WriteBuffer::type
  WriteBuffer::bufferType (void)
  {
    QDP_GUARD(lock_);
    return type_;
  }


  /**
   * Set buffer type
   */
  void
  WriteBuffer::setBufferType (WriteBuffer::type t)
  {
    QDP_GUARD(lock_);
    type_ = t;
  }

  /**
   * Set file offset for this buffer
   */
  void
  WriteBuffer::fileOffset (long fv)
  {
    QDP_GUARD(lock_);
    offset_ = fv;
  }


  /**
   * Get file offset for this buffer
   */
  long
  WriteBuffer::fileOffset (void)
  {
    QDP_GUARD(lock_);
    return offset_;
  }

  
  

  //=================================================================
  // Implementation of WriteBufferPool Class
  //=================================================================
  /**
   * constructor
   *
   * @param num_init_buffers the number of initial buffers to allocate
   * @param bufsize the size of each allocated buffer
   */
  WriteBufferPool::WriteBufferPool (int num_init_buffers, 
				    unsigned int bufsize)
    :wbuffers_(), fbuffers_(), bufsize_ (bufsize), monitor_()
  {
    for (int i = 0; i < num_init_buffers; i++) {
      WriteBuffer *buffer = new WriteBuffer (bufsize);
      wbuffers_.push_back (buffer);
    }
  }

  /**
   * Destructor
   */
  WriteBufferPool::~WriteBufferPool (void)
  {
    std::list<WriteBuffer *>::iterator wit;
    std::list<WriteBuffer *>::iterator fit;
#ifdef _DEBUG
    fprintf (stderr, "Cleaning buffers %d wbuffers and %d fbuffers.\n",
	     wbuffers_.size(), fbuffers_.size());
#endif
    for (wit = wbuffers_.begin(); wit != wbuffers_.end(); wit++)
      delete *wit;

    for (fit = fbuffers_.begin(); fit != fbuffers_.end(); fit++)
      delete *fit;
  }

  /**
   * Return a write buffer to copy data into for data of size 'nbytes'
   */
  WriteBuffer* 
  WriteBufferPool::specialBuffer (WriteBuffer::type t)
  {
    monitor_.lock();

    WriteBuffer* wbuf;
    if (wbuffers_.empty()) {  // we do not have any write buffer
      wbuf = new WriteBuffer (bufsize_);
#ifdef _DEBUG
      fprintf (stderr, "Creating a new special buffer.\n");
#endif
    }
    else {
      wbuf = wbuffers_.front();
      wbuffers_.pop_front ();
    }

    wbuf->setBufferType (t);
    // check whether there is flush buffer
    int numfbuf = fbuffers_.size();
    // move this buffer to the end of flush buffer list
    fbuffers_.push_back (wbuf);

    if (numfbuf == 0) { // the io thread is waiting
#ifdef _DEBUG
      fprintf (stderr, "Special No flushing buffer, notify.\n");
#endif
      monitor_.unlock();
      monitor_.notify ();
    }
    else {
#ifdef _DEBUG
      fprintf (stderr, "Special Has flushing buffer %d.\n", numfbuf);
#endif
      monitor_.unlock();
    }
    return wbuf;
  }

  /**
   * Get current write buffer
   */
  WriteBuffer* 
  WriteBufferPool::currentWriteBuffer (void)
  {
    WriteBuffer* wbuf = 0;
    monitor_.lock();
    
    if (wbuffers_.empty()) { // we do not have any write buffer
#ifdef _DEBUG
      fprintf (stderr, "Create a new buffer to hold file offset.\n");
#endif
      wbuffers_.push_back (new WriteBuffer (bufsize_));
    }

    wbuf = wbuffers_.front();
    monitor_.unlock();
    return wbuf;
  }

  /**
   * Return a write buffer to copy data into for data of size 'nbytes'
   */
  WriteBuffer* 
  WriteBufferPool::bufferToWrite (unsigned int nbytes)
  {
    WriteBuffer* wbuf = 0;
    monitor_.lock();
    
    if (wbuffers_.empty()) { // we do not have any write buffer
#ifdef _DEBUG
      fprintf (stderr, "There is no write buffer, create a new one.\n");
#endif
      wbuffers_.push_back (new WriteBuffer (bufsize_));
      wbuf = wbuffers_.front();
      monitor_.unlock();
      return wbuf;
    }
    else {
      wbuf = wbuffers_.front();
      if (wbuf->availableSpace () >= nbytes) {
	monitor_.unlock();
	return wbuf;
      }
      else {
	// check whether there is flush buffer
	int numfbuf = fbuffers_.size();
#ifdef _DEBUG
	fprintf (stderr, "Exisiting buffer has no space  flush = %d.\n",
		 numfbuf);
#endif

	// move this buffer to the end of flush buffer list
	fbuffers_.push_back (wbuf);

	wbuffers_.pop_front (); // remove this from wbuffer list
	if (wbuffers_.empty()) {
	  // we do not have any write buffer
#ifdef _DEBUG
	  fprintf (stderr, "No more write buffer, create a new one .\n");
#endif
	  wbuffers_.push_back (new WriteBuffer (bufsize_));
	}
	wbuf = wbuffers_.front();
	
	if (numfbuf == 0) { // the io thread is waiting
#ifdef _DEBUG
	  fprintf (stderr, "IO thread is waiting, notify. wbuffers = %d\n", wbuffers_.size ());
#endif
	  monitor_.unlock();
	  monitor_.notify ();
	}
	else {
#ifdef _DEBUG
	  fprintf (stderr, "IO threas is not waiting, no need to notify.\n");
#endif
	  monitor_.unlock();
	}
	return wbuf;
      }
    }
  }

  /**
   * Return a buffer to flush to disk. 
   * If there is no buffer to flush, this routine will block
   */
  WriteBuffer* 
  WriteBufferPool::bufferToFlush (void)
  {
    monitor_.lock();

    int numfbuf = fbuffers_.size ();
#ifdef _DEBUG
    fprintf (stderr, "Number of fbuffers = %d\n", numfbuf);
#endif
    while (fbuffers_.size() == 0) {
#ifdef _DEBUG
      fprintf (stderr, "IOThread is waiting !!!!!!!!!!!!!!!\n");
#endif
      monitor_.wait();
    }
#ifdef _DEBUG
    fprintf (stderr, "IOThread has a buffer now \n");
#endif
    WriteBuffer* fbuf = fbuffers_.front();
    monitor_.unlock();

    return fbuf;
  }

  /**
   * Release a flush buffer after this buffer is written to disk
   */
  void
  WriteBufferPool::releaseFlushBuffer (WriteBuffer* buffer)
  {
    monitor_.lock();

    // reset this buffer
    WriteBuffer::type t = buffer->bufferType();
    buffer->reset ();

    // this buffer is the first buffer of the flush buffer list
    fbuffers_.pop_front();

    // put this buffer back to wbuffers_
    wbuffers_.push_back (buffer);
    
#ifdef _DEBUG
    fprintf (stderr, "Put flush buffer type %d back to write pool into size %d. flush = %d\n",
	     t, wbuffers_.size(), fbuffers_.size());
#endif

    monitor_.unlock();
  }


  //=================================================================
  // Implementation of FileBufferWriter
  //=================================================================
  /**
   * Defaulr constructor
   * 
   * @param filename The file name to write data into
   * @param default_buffer_size The default buffer size to accumulate 
   * data into.
   * @param number_buffers The number of inital buffers to store data
   */
  FileBufferWriter::FileBufferWriter (std::ostream& fstream,
				      unsigned int default_buffer_size,
				      unsigned int number_buffers)
    :ioRunnable(), f(fstream),
     buffers_ (number_buffers, default_buffer_size), 
     bufsize_ (default_buffer_size),
     thread_(0), 
     flush_wait_(0), flush_monitor_()
  {
    // launch io write thread
    thread_ = ioThread::create (this, "qdpIOWriter");
  }
  
  /**
   * constructor
   * @param filename The file name to write data into
   * @param stripe_count The number of stripe count for the file
   * @param strip_size The stripe size for the file
   * @param number_buffers The number of inital buffers to store data
   */
  FileBufferWriter::FileBufferWriter (std::ostream& fstream,
				      unsigned int stripe_count,
				      unsigned int stripe_size,
				      unsigned int number_buffers)
    :ioRunnable(), f(fstream), 
     buffers_ (number_buffers, stripe_count * stripe_size),
     bufsize_ (stripe_count * stripe_size),
     thread_(0),
     flush_wait_(0), flush_monitor_()
  {
    // launch io write thread
    thread_ = ioThread::create (this, "qdpIOWriter");
  }

  /**
   * Destructor
   */
  FileBufferWriter::~FileBufferWriter (void)
  {
    // delete internal thread
    delete thread_;
    thread_= 0;
  }

  /**
   * Write data
   * @param data buffer of data to write
   * @param size the size of each element
   * @param numelem the number of elements each of which has size 'size'
   * @return returns 0 if everything is good, otherwise something is wrong
   */
  int 
  FileBufferWriter::write (char* data, size_t size, size_t numelem,
			   int swapbyte)
  {
    size_t totsize = numelem * size;

    if (totsize <= bufsize_) {
      // now copy data
      WriteBuffer* wbuf = buffers_.bufferToWrite (totsize);

      wbuf->copy ((unsigned char *)data, size, numelem, swapbyte);
      return 0;
    }
    else {
      // each single write size aligned with 'size'
      unsigned int swsize, swelem, numwrites;
      if (size <= bufsize_) {
	swelem = bufsize_/size;
	swsize = swelem * size;
	numwrites = numelem/swelem;
	if (numelem % swelem != 0)
	  numwrites ++;
	
	size_t nbytes = 0;
	size_t nelem  = 0;
	for (int i = 0; i < numwrites; i++) {
	  if (i != numwrites - 1) {
	    swrite (&(data[nbytes]), size, swelem, swapbyte);
	    nbytes += (size * swelem);
	    nelem += swelem;
	  }
	  else {
	    swrite (&(data[nbytes]), size, numelem - nelem, swapbyte);
	    nbytes += (size * (numelem - nelem));
	    nelem += (numelem - nelem);
	  }
	}
      }
      else {
	fprintf (stderr, "Single element size is too big to fit into a write buffer.\n");
	::exit (1);
      }
      return 0;
    }
  }

  /**
   * Set file offset
   */
  void
  FileBufferWriter::setFileOffset (long offset)
  {
    WriteBuffer* wbuf = buffers_.currentWriteBuffer ();
    wbuf->fileOffset (offset);
  }

  /**
   * IO output writer thread
   */
  void
  FileBufferWriter::run (void)
  {
    while (1) {
      // could be blocked here
      WriteBuffer* fbuf = buffers_.bufferToFlush ();

      if (fbuf->bufferType () == WriteBuffer::CLOSE_IO) {
	if (fbuf->dataSize () > 0) {
	  if (fbuf->fileOffset() != -1)
	    f.seekp (fbuf->fileOffset(), std::ios_base::beg);

	  f.write ((const char *)fbuf->data(), fbuf->dataSize());
	}

	buffers_.releaseFlushBuffer (fbuf);
	break;
      }
      else if (fbuf->bufferType () == WriteBuffer::FLUSH_IO) {
	if (fbuf->dataSize () > 0) {
	  if (fbuf->fileOffset() != -1)
	    f.seekp (fbuf->fileOffset(), std::ios_base::beg);

	  f.write ((const char *)fbuf->data(), fbuf->dataSize());
	}

	buffers_.releaseFlushBuffer (fbuf);

	// waker up flush waiter
	flush_monitor_.lock();
	if (flush_wait_ == 1) {
	  flush_wait_ = 0;
	  flush_monitor_.notify ();
	}
	flush_monitor_.unlock();
      }
      else {
	if (fbuf->fileOffset() != -1)
	  f.seekp (fbuf->fileOffset(), std::ios_base::beg);

	f.write ((const char *)fbuf->data(), fbuf->dataSize());

	buffers_.releaseFlushBuffer (fbuf);
      }

      // Now we need to check the status of write
      if (!f.good()) {
	fprintf (stderr, "Writing to file encountered error. \n");
	::exit (1);
      }
    }
  }

  /**
   * Close this writer
   */
  int
  FileBufferWriter::close (void)
  {
    // create a closing buffer
    WriteBuffer* wbuf = buffers_.specialBuffer (WriteBuffer::CLOSE_IO);

    // wait for thread to finish
    thread_->join ();
  }

  /**
   * Flush out all buffers and wait for it finish
   */
  int
  FileBufferWriter::flush (void)
  {
    flush_monitor_.lock ();
    flush_wait_ = 1;

    // create a flosu io buffer
    WriteBuffer* wbuf = buffers_.specialBuffer (WriteBuffer::FLUSH_IO);

    while (flush_wait_ != 0) {
      flush_monitor_.wait ();
    }

    flush_monitor_.unlock ();

    f.flush ();

#if 0
    // check the position
    long pos = f.tellp();
    fprintf (stderr, "Flush finished pos = %ld \n", pos);
    fflush (stderr);
#endif
  }

  /**
   * Write data
   * @param data buffer of data to write
   * @param size the size of each element
   * @param numelem the number of elements each of which has size 'size'
   * @return returns 0 if everything is good, otherwise something is wrong
   */
  int 
  FileBufferWriter::swrite (char* data, size_t size, size_t numelem,
			    int swapbyte)
  {
    // now copy data
    WriteBuffer* wbuf = buffers_.bufferToWrite (size * numelem);

    wbuf->copy ((unsigned char *)data, size, numelem, swapbyte);
  }

}

