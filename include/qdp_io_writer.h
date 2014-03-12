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
 * ---------------------------------------------------------------------
 * Description:
 *     QDP++ I/O Writer Thread and related routines
 *
 *     The purpose of this class is to buffer QDP binary file writer output
 *     and to write to underlying global file system, which mostlikely
 *     to be a Lustre file system
 *     
 *
 * Author:
 *     Jie Chen
 *     Scientific Computing Group
 *     Jefferson Lab
 *
 * Revision History:
 *     $Log:  $
 *
 */
#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <pthread.h>
#include <list>
#include <iostream>
#include <fstream>
#include "qdp_io_thread.h"

namespace QDPIOThread
{
  /**
   * An object of this class will not be used by more than
   * one thread at a time. So no mutex is needed
   */
  class WriteBuffer
  {
  public:
    /**
     * Buffer type
     */
    enum type {REGULAR=1000, FLUSH_IO, CLOSE_IO};

    /**
     * constructor
     */
    WriteBuffer (unsigned int size);

    /**
     * Destructor
     */
    ~WriteBuffer (void);

    /**
     * Return data pointer
     */
    unsigned char* data (void) const;

    /**
     * Return current size of data
     */
    unsigned int dataSize (void);


    /**
     * Copy data into this buffer with option of swapping bytes
     */
    void copy (unsigned char* data, size_t size, size_t nelem, 
	       int swapbytes = 1);

    /**
     * Return current space for data
     */
    unsigned int availableSpace (void);

    /**
     * Return total size of buffer
     */
    unsigned int size (void) const;

    /**
     * Reset this buffer to empty
     */
    void reset (void);

    /**
     * buffer type
     */
    WriteBuffer::type bufferType (void);

    /**
     * set closing buffer flag
     */
    void setBufferType (WriteBuffer::type t);

    /**
     * get current offset value
     */
    long fileOffset (void);

    /**
     * Set file offset value
     */
    void fileOffset (long fv);


  private:
    /* write buffer size              */
    unsigned int bufsize_;

    /* accumulated data size          */
    unsigned int datasize_;

    /* offset value inside of a file  */
    /* if this value = -1, no need to */
    /* to do seek                     */
    long offset_;

    /* real buffer                    */
    unsigned char *data_;

    /* special finishing buffer       */
    /* if this flag is on, this is the last buffer to write */
    WriteBuffer::type type_;

    /* mutex to protect               */
    ioMutex lock_;

    /* hide copy and assignment operators */
    WriteBuffer (const WriteBuffer& buffer);
    WriteBuffer& operator = (const WriteBuffer& buffer);
  };

  class WriteBufferPool
  {
  public:
    /**
     * constructor
     *
     * @param num_init_buffers the number of initial buffers to allocate
     * @param bufsize the size of each allocated buffer
     */
    WriteBufferPool (int num_init_buffers, unsigned int bufsize);

    /**
     * Destructor
     */
    ~WriteBufferPool (void);

    /**
     * Send a special buffer either closing or flushing to IO thread
     */
    WriteBuffer* specialBuffer (WriteBuffer::type t = WriteBuffer::CLOSE_IO);

    /**
     * Return a write buffer to copy data into for data of size 'nbytes'
     */
    WriteBuffer* bufferToWrite (unsigned int nbytes);

    /**
     * Get current write buffer
     */
    WriteBuffer* currentWriteBuffer (void);


    /**
     * Return a buffer to flush to disk. 
     * If there is no buffer to flush, this routine will block
     */
    WriteBuffer* bufferToFlush (void);

    /**
     * Release a flush buffer back to write buffer list
     */
    void releaseFlushBuffer (WriteBuffer* buffer);

  private:
    /* buffer queue                   */
    std::list<WriteBuffer *> wbuffers_;
    std::list<WriteBuffer *> fbuffers_;

    /* keep track default buffer size */
    unsigned int bufsize_;

    /* monitor object for non-empty flush buffer */
    ioMonitor monitor_;

    /* hide copy and assignment operators */
    WriteBufferPool (const WriteBufferPool& pool);
    WriteBufferPool& operator = (const WriteBufferPool& pool);
  };

  class FileBufferWriter : public ioRunnable
  {
  public:
    
    /**
     * Defaulr constructor
     * 
     * @param default_buffer_size The default buffer size to accumulate 
     * data into.
     * @param number_buffers The number of inital buffers to store data
     */
    FileBufferWriter (std::ostream& fstream,
		      unsigned int default_buffer_size,
		      unsigned int number_buffers);

    /**
     * constructor
     * @param stripe_count The number of stripe count for the file
     * @param strip_size The stripe size for the file
     * @param number_buffers The number of inital buffers to store data
     */
    FileBufferWriter (std::ostream& fstream,
		      unsigned int stripe_count,
		      unsigned int stripe_size,
		      unsigned int number_buffers);

    /**
     * Destructor
     */
    ~FileBufferWriter (void);

    /**
     * Inherited member function from ioRunnable
     */
    void run (void);

    /**
     * Write data
     * @param data buffer of data to write
     * @param size the size of each element
     * @param numelem the number of elements each of which has size 'size'
     * @param swapbyte do we swap byte? default is yes
     * @return returns 0 if everything is good, otherwise something is wrong
     */
    int write (char* data, size_t size, size_t numelem, int swapbyte=1);


    /**
     * Set file offset
     */
    void setFileOffset (long offset);

    /**
     * Close the file buffer and wait
     */
    int close (void);

    /**
     * Flush all buffers and wait
     */
    int flush (void);

  private:
    /* file stream                   */
    std::ostream& f;

    /* underlying buffer pool        */
    WriteBufferPool buffers_;

    /* buffer size                   */
    unsigned int bufsize_;

    /* underlying thread doing I/O   */
    ioThread* thread_;

    /* waiting for io thread to flush out */
    int flush_wait_;

    /* monitor object to notify a thread waiting on flush */
    ioMonitor flush_monitor_;    

    /* internal data write routine. Each write has data size smaller
     * than buffer size
     */
    int swrite (char* data, size_t size, size_t numelem, int swapbyte=1);


    /* hide copy and assignment      */
    FileBufferWriter (const FileBufferWriter& );
    FileBufferWriter& operator = (const FileBufferWriter& );
  };
}

