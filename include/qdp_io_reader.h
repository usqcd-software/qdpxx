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
 *     QDP++ I/O Reader Thread and related routines
 *
 *     The purpose of this class is to read a binary file and put the
 *     content into buffers. The regular reading thread read from
 *     the buffers.
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
#include <vector>
#include "qdp_io_thread.h"

namespace QDPIOThread
{
  /**
   * An object of this class will not be used by more than
   * one thread at a time. So no mutex is needed
   */
  class RWBuffer
  {
  public:
    /**
     * Buffer type
     */
    enum type {REGULAR=1000, FLUSH_IO, CLOSE_IO};

    /**
     * constructor
     */
    RWBuffer (unsigned int size);

    /**
     * Destructor
     */
    ~RWBuffer (void);

    /**
     * Return read data pointer
     */
    unsigned char* readData (void);


    /**
     * Return write data pointer
     */
    unsigned char* writeData (void);


    /**
     * Copy data from this buffer to another allocated buffer
     */
    void copy (unsigned char* dest, size_t size, size_t nelem, 
	       int swapbytes = 1);

    /**
     * Return total size of buffer
     */
    unsigned int size (void) const;

    /**
     * Reset this buffer to empty
     */
    void reset (void);
    
    /**
     * Available read space
     */
    unsigned int availableReadData (void);


    /**
     * Available write space
     */
    unsigned int availableWriteSpace (void);

    /**
     * Decrease write space by this many bytes
     */
    void descreaseWriteSpace (unsigned int nbytes);

    /**
     * buffer type
     */
    RWBuffer::type bufferType (void);

    /**
     * set closing buffer flag
     */
    void setBufferType (RWBuffer::type t);

    /**
     * get current offset value
     */
    long fileOffset (void);

    /**
     * Set file offset value
     */
    void fileOffset (long fv);

    /**
     * Get next buffer 
     */
    RWBuffer* nextBuffer (void);

  private:
    /* read buffer size               */
    unsigned int bufsize_;

    /* offset value inside of a file  */
    /* if this value = -1, no need to */
    /* to do seek                     */
    long offset_;

    /* real buffer                    */
    unsigned char *data_;

    /* current read index             */
    unsigned int rd_idx_;

    /* write index                    */
    unsigned int wr_idx_;

    /* special finishing buffer       */
    /* if this flag is on, this is the last buffer to write */
    RWBuffer::type type_;

    /* mutex to protect               */
    ioMutex lock_;

    /* next buffer                    */
    RWBuffer* next_;

    /* hide copy and assignment operators */
    RWBuffer (const RWBuffer& buffer);
    RWBuffer& operator = (const RWBuffer& buffer);
  };

  class RWBufferPool
  {
  public:
    /**
     * constructor
     *
     * @param num_init_buffers the number of initial buffers to allocate
     * @param bufsize the size of each allocated buffer
     * @param max_read_bytes maximum number of bytes to read
     */
    RWBufferPool (int num_init_buffers, unsigned int bufsize,
		  unsigned long max_read_bytes);

    /**
     * Destructor
     */
    ~RWBufferPool (void);

    /**
     * Send a special buffer either closing or flushing to IO thread
     */
    void insertSpecialBuffer (RWBuffer::type t = RWBuffer::CLOSE_IO);

    /**
     * Copy data to destination buffer
     */
    int copyToDest (char* dest, size_t size, size_t numelem, int swapbyte = 1);


    /**
     * Release a copy buffer back to read file buffer list
     */
    void releaseCopyBuffer (RWBuffer* buffer);


    /**
     * Release a file buffer back to copy buffer list
     */
    void releaseFileBuffer (RWBuffer* buffer);


    /**
     * Return a write buffer to read file into
     *
     */
    RWBuffer* fileBuffer (void);

    /**
     * Return total number of bytes in memory
     */
    unsigned long totalMemory (void);


  private:
    /* buffer queue                   */
    std::list<RWBuffer *> copy_buffers_;
    std::list<RWBuffer *> file_buffers_;

    /* keep track default buffer size */
    unsigned int bufsize_;

    /* maximum number of bytes to read */
    unsigned long max_read_bytes_;

    /* monitor object for non-empty copy buffer */
    ioMonitor monitor_;

    /* hide copy and assignment operators */
    RWBufferPool (const RWBufferPool& pool);
    RWBufferPool& operator = (const RWBufferPool& pool);
  };

  class FileBufferReader : public ioRunnable
  {
  public:
    
    /**
     * Defaulr constructor
     * 
     * @param default_buffer_size The default buffer size to accumulate 
     * data into.
     * @param number_buffers The number of inital buffers to store data
     * @param max_read_bytes maximum number of bytes to read
     * @param elem_size the size of each element
     */
    FileBufferReader (std::istream& fstream,
		      unsigned int default_buffer_size,
		      unsigned int number_buffers,
		      unsigned long max_read_bytes,
		      unsigned int elem_size);

    /**
     * constructor
     * @param stripe_count The number of stripe count for the file
     * @param strip_size The stripe size for the file
     * @param number_buffers The number of inital buffers to store data
     * @param max_read_bytes maximum number of bytes to read
     * @param elem_size the size of each element
     */
    FileBufferReader (std::istream& fstream,
		      unsigned int stripe_count,
		      unsigned int stripe_size,
		      unsigned int number_buffers,
		      unsigned long max_read_bytes,
		      unsigned int elem_size);

    /**
     * Destructor
     */
    ~FileBufferReader (void);

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
    int read (char* dest, size_t size, size_t numelem, int swapbyte=1);


    /**
     * Set file offset
     */
    void setFileOffset (long offset);

    /**
     * Close the copy buffer and wait
     */
    int close (void);

  private:
    /* file stream                   */
    std::istream& f;

    /* underlying buffer pool        */
    RWBufferPool buffers_;

    /* max number of bytes to read   */
    unsigned long max_read_bytes_;

    /* element size                  */
    unsigned int elem_size_;

    /* number of read bytes so far   */
    unsigned long read_bytes_;

    /* underlying thread doing I/O   */
    ioThread* thread_;

    /* waiting for io thread to flush out */
    int flush_wait_;

    /* monitor object to notify a thread waiting on flush */
    ioMonitor flush_monitor_;    


    /* hide copy and assignment      */
    FileBufferReader (const FileBufferReader& );
    FileBufferReader& operator = (const FileBufferReader& );
  };


  /**
   * Read sub block information
   * This class is for qdp only
   */
  class RdSubBlkInfo
  {
  public:
    // default constructor
    RdSubBlkInfo (void);

    // constructor
    RdSubBlkInfo (size_t ssize, unsigned int num_xinc, int startingSite);
    // copy constructor
    RdSubBlkInfo (const RdSubBlkInfo& info);
    // assignment operator
    RdSubBlkInfo& operator = (const RdSubBlkInfo& info);
    
    // Destructor
    ~RdSubBlkInfo (void);

    // Get and Set for variables
    size_t SubBlockSize (void) const;
    void SubBlockSize (size_t size);

    unsigned int NumXIncrements (void) const;
    void NumXIncrements (unsigned int num);

    int StartingSite (void) const;
    void StartingSite (int s);

  private:
    // size of this sub blocks
    size_t sub_blk_size_;
    // number of x increments contained in the sub blocks
    unsigned int num_xinc_;
    // starting site number
    int starting_site_;
  };

  /**
   * Read Data Block Information
   * This class is for qdp only
   */
  class RdBlkInfo
  {
  public:
    // default constructor
    RdBlkInfo (void);

    // constructor
    RdBlkInfo (int rank, size_t totalSize, int startingSite,
	       size_t xincSize, int xincSites, size_t maxBufSize);

    // copy constructor
    RdBlkInfo (const RdBlkInfo& info);

    // assignment operator
    RdBlkInfo& operator = (const RdBlkInfo& info);

    // destructor
    ~RdBlkInfo (void);

    // Set up information
    void SetupInfo (int rank, size_t totalSize, int startingSite,
		    size_t xincSize, int xincSites, size_t maxBufSize);

    // return sub block information directly
    unsigned int NumberSubBlocks (void) const;
    size_t SubBlockSize (int i) const;
    unsigned int SubBlockXIncrements (int i) const;
    int SubBlockStartingSite (int i) const;

    int XincSites (void) const;
    void XincSites (int x);

    int NodeNumber (void) const;
    void NodeNumber (int n);

  private:
    // mpi rank for this read block
    int rank_;

    // total size of this block
    size_t total_size_;

    // sub blocks
    std::vector<RdSubBlkInfo> sub_blocks_;

    // starting site of this block
    int starting_site_;

    // size of bytes of each x-increment
    size_t xinc_tot_size_;

    // number of site each xinc has
    int xinc_sites_;

    // maximum size for sub_block size
    size_t max_sub_blksize_;

    // set up sub block information
    void SetupSubBlocks (void);

    // implement standard output stream
    friend std::ostream& operator << (std::ostream& out, const RdBlkInfo& info);
  };

}

