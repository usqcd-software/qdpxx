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
 *     QDP++ I/O Reader Thread and related routines
 *     
 *
 * Author:
 *     Jie Chen
 *     Scientific Computing Group
 *     Jefferson Lab
 *
 */
#include "qdp_io_reader.h"
#include <stdlib.h>
#include <byteswap.h>

using namespace std;

namespace QDPIOThread
{ 
  //=================================================================
  // Implementation of RWBuffer Class
  //=================================================================

  /**
   * constructor
   */
  RWBuffer::RWBuffer (unsigned int size)
    :bufsize_(size), offset_(-1), 
     data_(0), rd_idx_ (0), wr_idx_(0), type_ (RWBuffer::REGULAR), 
     next_(0), lock_()
  {
    // allocate space aligned on unsigned long
    // C++ makes sure tmp is aligned on unsigned long
    unsigned long *tmp = new unsigned long[size/sizeof(unsigned long) + 1];
    data_ = (unsigned char *)tmp;
  }

  /**
   * Destructor
   */
  RWBuffer::~RWBuffer (void)
  {
    delete []data_;
    next_ = 0;
  }

  /**
   * Return total size of buffer
   */
  unsigned int 
  RWBuffer::size (void) const
  {
    return bufsize_;
  }


  /**
   * Return current read data pointer
   */
  unsigned char* 
  RWBuffer::readData (void)
  {
    return &data_[rd_idx_];
  }

  /**
   * Return current write data pointer
   */
  unsigned char* 
  RWBuffer::writeData (void)
  {
    return &data_[wr_idx_];
  }

  /**
   * Reset buffer to initial state
   */
  void
  RWBuffer::reset (void)
  {
    QDP_GUARD(lock_);
    type_ = RWBuffer::REGULAR;
    offset_ = -1;
    rd_idx_ = 0;
    wr_idx_ = 0;
    next_ = 0;
  }

  
  /**
   * Return available readable data
   */
  unsigned int 
  RWBuffer::availableReadData (void)
  {
    QDP_GUARD(lock_);
    return wr_idx_ - rd_idx_;
  }


  /**
   * Return available writeable space
   */
  unsigned int 
  RWBuffer::availableWriteSpace (void)
  {
    QDP_GUARD(lock_);
    return bufsize_ - wr_idx_;
  }

  /**
   * Decrease write space by this many bytes
   */
  void 
  RWBuffer::descreaseWriteSpace (unsigned int nbytes)
  {
    QDP_GUARD(lock_);
    wr_idx_ += nbytes;
  }

  /**
   * Copy data into this buffer with option of swapping bytes
   */
  void
  RWBuffer::copy (unsigned char* dest, size_t size, size_t nelem, 
		  int swapbytes)
  {
    QDP_GUARD(lock_);
    if (swapbytes) {
      switch (size) {
      case 4:  // float or integer
	{
	  unsigned int *s = (unsigned int *)&(data_[rd_idx_]);
	  unsigned int *d = (unsigned int *)dest;

	  for(int j = 0; j < nelem; j++)
	    d[j] = bswap_32(s[j]);	    
	}
	break;
      case 8:  /* n_uint64_t or double */
	{
	  unsigned long *s = (unsigned long *)&(data_[rd_idx_]);
	  unsigned long *d = (unsigned long *)dest;

	  for(int j = 0; j < nelem; j++)
	    d[j] = bswap_64(s[j]);	  
	}
	break;
      case 2:  /* n_uint16_t */
	{
	  unsigned short *s = (unsigned short *)&(data_[rd_idx_]);
	  unsigned short *d = (unsigned short *)dest;

	  for(int j = 0; j < nelem; j++)
	    d[j] = bswap_16(s[j]);
	}
	break;
      case 1:  /* n_uint8_t: byte - do nothing */
	memcpy (dest, &(data_[rd_idx_]), size * nelem);
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
      memcpy (dest, &(data_[rd_idx_]),size * nelem);

    rd_idx_ += (size * nelem);
  }

  /**
   * check this buffer type
   */
  RWBuffer::type
  RWBuffer::bufferType (void)
  {
    QDP_GUARD(lock_);
    return type_;
  }


  /**
   * Set buffer type
   */
  void
  RWBuffer::setBufferType (RWBuffer::type t)
  {
    QDP_GUARD(lock_);
    type_ = t;
  }

  /**
   * Set file offset for this buffer
   */
  void
  RWBuffer::fileOffset (long fv)
  {
    QDP_GUARD(lock_);
    offset_ = fv;
  }


  /**
   * Get file offset for this buffer
   */
  long
  RWBuffer::fileOffset (void)
  {
    QDP_GUARD(lock_);
    return offset_;
  }

  
  //=================================================================
  // Implementation of RWBufferPool Class
  //=================================================================
  /**
   * constructor
   *
   * @param num_init_buffers the number of initial buffers to allocate
   * @param bufsize the size of each allocated buffer
   * @param max_read_bytes maximum number of bytes to read
   */
  RWBufferPool::RWBufferPool (int num_init_buffers, unsigned int bufsize,
			      unsigned long max_read_bytes)
    :copy_buffers_(), file_buffers_(), bufsize_(bufsize),
     max_read_bytes_ (max_read_bytes), monitor_()
  {
    for (int i = 0; i < num_init_buffers; i++) {
      RWBuffer *buffer = new RWBuffer (bufsize);
      file_buffers_.push_back (buffer);
    }
  }

  /**
   * Destructor
   */
  RWBufferPool::~RWBufferPool (void)
  {
    std::list<RWBuffer *>::iterator cit;
    std::list<RWBuffer *>::iterator fit;
#ifdef _DEBUG
    fprintf (stderr, "Cleaning buffers %d copy buffers and %d file buffers.\n",
	     copy_buffers_.size(), file_buffers_.size());
#endif

    for (fit = file_buffers_.begin(); fit != file_buffers_.end(); fit++)
      delete *fit;

    for (cit = copy_buffers_.begin(); cit != copy_buffers_.end(); cit++)
      delete *cit;
  }

  /**
   * Inject a special buffer to signal end of I/O
   */
  void
  RWBufferPool::insertSpecialBuffer (RWBuffer::type t)
  {
    monitor_.lock();

    RWBuffer* fbuf;
    if (file_buffers_.empty()) {  // we do not have any write buffer
      fbuf = new RWBuffer (bufsize_);
      file_buffers_.push_back (fbuf);
    }
    fbuf = file_buffers_.front();

    fbuf->setBufferType (t);

    monitor_.unlock();
  }



  /**
   * Return the number of bytes copied
   */
  int
  RWBufferPool::copyToDest (char* dest, size_t size, size_t numelem, int swapbyte)
  {
    unsigned int nbytes = size * numelem;
    unsigned int bytes_copied = 0;
    unsigned int didx = 0;
    RWBuffer *curr = 0;
    unsigned int num_elem_read, rem_nbytes;

    rem_nbytes = nbytes;

    while (rem_nbytes > 0) {
      monitor_.lock();

      // get a copy buffer
      while (copy_buffers_.size() == 0) 
	monitor_.wait ();

      // now we have a copy buffer
      curr = copy_buffers_.front ();

#ifdef _DEBUG
      fprintf (stderr, "main: available data = %d rem bytes = %d\n",
	       curr->availableReadData(), rem_nbytes);
#endif

      // check how much space one can read out from this buffer
      if (curr->availableReadData() >= rem_nbytes) {
	// release lock
	monitor_.unlock();

	// figure out how many elements to read
	num_elem_read = rem_nbytes/size;
	// copy to destination
	curr->copy ((unsigned char *)&dest[didx], size, num_elem_read, swapbyte);
	// available read data reduced in the above call
	didx += (size * num_elem_read);
	bytes_copied += (size * num_elem_read);
	rem_nbytes -= (size * num_elem_read);
      }
      else {
	// remove this buffer from copy buffer list
	copy_buffers_.pop_front ();

	// release lock
	monitor_.unlock();

	// figure out how many elements to read
	num_elem_read = curr->availableReadData()/size;

#ifdef _DEBUG
	fprintf (stderr, "main: num elems = %d rem_bytes = %d\n", num_elem_read,
		 rem_nbytes);
#endif
	// copy partial data into destination
	if (num_elem_read > 0) {
	  curr->copy ((unsigned char *)&dest[didx], size, num_elem_read, swapbyte);
	  // increase destination index
	  didx += (num_elem_read * size);
	  bytes_copied += (size * num_elem_read);
	  // how many bytes left
	  rem_nbytes -= (size * num_elem_read);
	}
	// now we need to put this buffer to file copy buffer
	// this routine has lock
	releaseCopyBuffer (curr);
      }
    }
    return bytes_copied;
  }

  /**
   * Return a file buffer to read from a file
   */
  RWBuffer* 
  RWBufferPool::fileBuffer (void)
  {
    monitor_.lock();
    RWBuffer* fbuf = 0;

    // check whether we are empty or not
    if (file_buffers_.size() == 0) {
      fbuf = new RWBuffer (bufsize_);
      file_buffers_.push_back (fbuf);
    }
    
    // now we should have a buffer
    fbuf = file_buffers_.front ();
    monitor_.unlock();

    return fbuf;
  }

  /**
   * Release a copy buffer after this buffer is copied out
   */
  void
  RWBufferPool::releaseCopyBuffer (RWBuffer* buffer)
  {
    monitor_.lock();

    // reset this buffer
    buffer->reset ();

    // this buffer is the first buffer of the flush buffer list
    // copy_buffers_.pop_front();

    // put this buffer back to file buffers_
    file_buffers_.push_back (buffer);
    
#ifdef _DEBUG
    fprintf (stderr, "Put copy buffer back to file buffer pool from size %d file buffers = %d\n",
	     copy_buffers_.size(), file_buffers_.size());
#endif

    monitor_.unlock();
  }


  /**
   * Release a flush buffer after this buffer is written to disk
   */
  void
  RWBufferPool::releaseFileBuffer (RWBuffer* buffer)
  {
    monitor_.lock();

    int num_copy_bufs = copy_buffers_.size();

    // this buffer is the first buffer of the flush buffer list
    file_buffers_.pop_front();

    // put this buffer back to wbuffers_
    copy_buffers_.push_back (buffer);
    
#ifdef _DEBUG
    fprintf (stderr, "Put file buffer to copy pool from size %d. copy buffers = %d\n",
	     file_buffers_.size(), copy_buffers_.size());
#endif

    if (num_copy_bufs == 0) {
      monitor_.notify ();
      monitor_.unlock ();
    }
    else
      monitor_.unlock();
  }

  /**
   * check total memory used in the buffer pool
   */
  unsigned long
  RWBufferPool::totalMemory (void)
  {
    monitor_.lock();

    unsigned long bytes = file_buffers_.size() * bufsize_ + copy_buffers_.size() * bufsize_;

    monitor_.unlock();
    return bytes;
  }


  //=================================================================
  // Implementation of FileBufferWriter
  //=================================================================
  /**
   * Default constructor
   * 
   * @param filename The file name to write data into
   * @param default_buffer_size The default buffer size to accumulate 
   * data into.
   * @param number_buffers The number of inital buffers to store data
   * @param max_read_bytes maximum number of bytes to read
   */
  FileBufferReader::FileBufferReader (std::istream& input,
				      unsigned int default_buffer_size,
				      unsigned int number_buffers,
				      unsigned long max_read_bytes,
				      unsigned int elem_size)
    :ioRunnable(), f(input), 
     buffers_ (number_buffers, default_buffer_size, max_read_bytes), 
     max_read_bytes_(max_read_bytes), elem_size_ (elem_size),
     read_bytes_(0), thread_(0), flush_wait_(0), flush_monitor_()
  {
    // launch io write thread
    thread_ = ioThread::create (this, "qdpIOReader");
  }
  
  /**
   * constructor
   * @param filename The file name to write data into
   * @param stripe_count The number of stripe count for the file
   * @param strip_size The stripe size for the file
   * @param number_buffers The number of inital buffers to store data
   * @param max_read_bytes maximum number of bytes to read
   */
  FileBufferReader::FileBufferReader (std::istream& input,
				      unsigned int stripe_count,
				      unsigned int stripe_size,
				      unsigned int number_buffers,
				      unsigned long max_read_bytes,
				      unsigned int elem_size)
    :ioRunnable(), f(input), 
     buffers_ (number_buffers, stripe_count * stripe_size, max_read_bytes), 
     max_read_bytes_(max_read_bytes), elem_size_ (elem_size), 
     read_bytes_(0), thread_(0), flush_wait_(0), flush_monitor_()
  {
    // launch io write thread
    thread_ = ioThread::create (this, "qdpIOReader");
  }

  /**
   * Destructor
   */
  FileBufferReader::~FileBufferReader (void)
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
  FileBufferReader::read (char* dest, size_t size, size_t numelem,
			  int swapbyte)
  {
    // now copy data
    return buffers_.copyToDest (dest, size, numelem, swapbyte);
  }


  /**
   * Set file offset
   */
  void
  FileBufferReader::setFileOffset (long offset)
  {
    RWBuffer* fbuf = buffers_.fileBuffer ();
    fbuf->fileOffset (offset);
  }

  /**
   * IO output writer thread
   */
  void
  FileBufferReader::run (void)
  {
    unsigned long nbytes_to_read, nbytes_left;

#ifdef _DEBUG
    fprintf (stderr, "io thread started readbytes = %lu maxbytes = %lu.\n",
	     read_bytes_, max_read_bytes_);
#endif

    while (read_bytes_ < max_read_bytes_) {
#ifdef _DEBUG
      fprintf (stderr, "iothread: read_bytes = %lu and max_bytes = %lu\n",
	       read_bytes_, max_read_bytes_);
#endif
      // file buffer
      RWBuffer* fbuf = buffers_.fileBuffer ();

      if (fbuf->fileOffset() != -1)
	f.seekg (fbuf->fileOffset(), std::ios_base::beg);

      // figure out how much space this buffer can be written to
      // we need to keep an element intact in a buffer
      nbytes_left = max_read_bytes_ - read_bytes_;
      if (fbuf->availableWriteSpace() <= nbytes_left) 
	nbytes_to_read = fbuf->availableWriteSpace()/elem_size_ * elem_size_;
      else
	nbytes_to_read = nbytes_left/elem_size_ * elem_size_;

      // read data into buffer
      f.read ((char *)fbuf->writeData(), nbytes_to_read);

      // quick check data
      int *data = (int *)fbuf->readData();

      if (f.good()) {
	read_bytes_ += nbytes_to_read;
	fbuf->descreaseWriteSpace (nbytes_to_read);
      }
      else {
	fprintf (stderr, "Fatal error: Reading data from a file encountered error. \n");
	::exit (1);
      }

      // when this buffer is full or the final data is read
      if (fbuf->availableWriteSpace() < elem_size_ || read_bytes_ >= max_read_bytes_)
	buffers_.releaseFileBuffer (fbuf);

      // thread is going to quit
      if (fbuf->bufferType() == RWBuffer::CLOSE_IO)
	break;
    }

  }

  /**
   * Close this writer
   */
  int
  FileBufferReader::close (void)
  {
    // create a closing buffer
    buffers_.insertSpecialBuffer (RWBuffer::CLOSE_IO);

    // wait for thread to finish
    thread_->join ();

#if 0
    fprintf (stderr, "IOThread finished using %lu bytes of memory.\n",
	     buffers_.totalMemory());
#endif
  }


  //=======================================================================
  // Implementation of SubBlockReadInfo Classes
  //=======================================================================
  RdSubBlkInfo::RdSubBlkInfo (void)
    :sub_blk_size_(0), num_xinc_(0), starting_site_(0)
  {
    // empty
  }

  RdSubBlkInfo::RdSubBlkInfo (size_t ssize, unsigned int num_xinc, 
			      int startingSite)
    :sub_blk_size_ (ssize), num_xinc_(num_xinc), starting_site_(startingSite)
  {
    // empty
  }

  RdSubBlkInfo::RdSubBlkInfo (const RdSubBlkInfo& sinfo)
    :sub_blk_size_ (sinfo.sub_blk_size_), num_xinc_(sinfo.num_xinc_),
     starting_site_ (sinfo.starting_site_)
  {
    // empty
  }

  RdSubBlkInfo&
  RdSubBlkInfo::operator = (const RdSubBlkInfo& sinfo)
  {
    if (this != &sinfo) {
      sub_blk_size_ = sinfo.sub_blk_size_;
      num_xinc_ = sinfo.num_xinc_;
      starting_site_ = sinfo.starting_site_;
    }
    return *this;
  }

  RdSubBlkInfo::~RdSubBlkInfo (void)
  {
    // empty
  }
  
  size_t
  RdSubBlkInfo::SubBlockSize (void) const
  {
    return sub_blk_size_;
  }

  void
  RdSubBlkInfo::SubBlockSize (size_t size)
  {
    sub_blk_size_ = size;
  }

  unsigned int
  RdSubBlkInfo::NumXIncrements (void) const
  {
    return num_xinc_;
  }

  void
  RdSubBlkInfo::NumXIncrements (unsigned int num)
  {
    num_xinc_ = num;
  }


  int 
  RdSubBlkInfo::StartingSite (void) const
  {
    return starting_site_;
  }

  void 
  RdSubBlkInfo::StartingSite (int s)
  {
    starting_site_ = s;
  }

  //===========================================================
  //       Implementation of Read Block Information
  //===========================================================
  RdBlkInfo::RdBlkInfo (void)
    :rank_(-1), total_size_ (0), sub_blocks_(), starting_site_(0),
     xinc_tot_size_(0), xinc_sites_(0), max_sub_blksize_(0)
  {
    // empty
  }

  RdBlkInfo::RdBlkInfo (int rank, size_t totalSize, int startingSite,
			size_t xincSize, int xincSites, size_t maxBufSize)
    :rank_(rank), total_size_ (totalSize), sub_blocks_(), 
     starting_site_(startingSite), xinc_tot_size_(xincSize), 
     xinc_sites_ (xincSites), max_sub_blksize_(maxBufSize)
  {
    SetupSubBlocks ();
  }

  // copy constructor
  RdBlkInfo::RdBlkInfo (const RdBlkInfo& info)
    :rank_(info.rank_), total_size_ (info.total_size_), sub_blocks_(), 
     starting_site_(info.starting_site_), xinc_tot_size_(info.xinc_tot_size_), 
     xinc_sites_ (info.xinc_sites_), max_sub_blksize_(info.max_sub_blksize_)
  {
    for (int i = 0; i < info.sub_blocks_.size(); i++)
      sub_blocks_.push_back (info.sub_blocks_[i]);
  }

  // assignment operator
  RdBlkInfo& 
  RdBlkInfo::operator = (const RdBlkInfo& info)
  {
    if (this != &info) {
      rank_ = info.rank_;
      total_size_  = info.total_size_;
      starting_site_ = info.starting_site_;
      xinc_tot_size_ = info.xinc_tot_size_;
      xinc_sites_ = info.xinc_sites_;
      max_sub_blksize_ = info.max_sub_blksize_;
      sub_blocks_.clear ();
      for (int i = 0; i < info.sub_blocks_.size(); i++)
	sub_blocks_.push_back (info.sub_blocks_[i]);
    }
    return *this;
  }

  RdBlkInfo::~RdBlkInfo (void)
  {
    // empty
  }

  void
  RdBlkInfo::SetupInfo (int rank, size_t totalSize, int startingSite,
			size_t xincSize, int xincSites, size_t maxBufSize)
  {
    rank_ = rank;
    total_size_ = totalSize;
    starting_site_ = startingSite;
    xinc_tot_size_ = xincSize;
    xinc_sites_ = xincSites;
    max_sub_blksize_ = maxBufSize;
    sub_blocks_.clear ();

    SetupSubBlocks ();
  }

  void
  RdBlkInfo::SetupSubBlocks (void)
  {
    // if the total size is bigger than max buf size, we need to
    // split the block into multiple sub blocks
    if (total_size_ <= max_sub_blksize_) {
      RdSubBlkInfo sinfo (total_size_, total_size_/xinc_tot_size_, 
			  starting_site_);
      sub_blocks_.push_back (sinfo);
    }
    else {
      int ssite = 0;
      size_t sblksize = max_sub_blksize_/xinc_tot_size_ * xinc_tot_size_;
      unsigned int num_sub_blocks = total_size_/sblksize;
      if (total_size_ % sblksize != 0)
	num_sub_blocks++;
      
      for (int i = 0; i < num_sub_blocks; i++) {
	ssite = starting_site_ + i * (sblksize/xinc_tot_size_) * xinc_sites_;
	if (i != num_sub_blocks - 1) {
	  RdSubBlkInfo sinfo (sblksize, sblksize/xinc_tot_size_, ssite);
	  sub_blocks_.push_back (sinfo);
	}
	else {
	  sblksize = total_size_ - (num_sub_blocks - 1) * sblksize;
	  RdSubBlkInfo sinfo (sblksize, sblksize/xinc_tot_size_, ssite);  
	  sub_blocks_.push_back (sinfo);
	}
      }
    }
  }
  

  unsigned int
  RdBlkInfo::NumberSubBlocks (void) const
  {
    return sub_blocks_.size();
  }

  size_t
  RdBlkInfo::SubBlockSize (int i) const
  {
    return sub_blocks_[i].SubBlockSize();
  }

  unsigned int
  RdBlkInfo::SubBlockXIncrements (int i) const
  {
    return sub_blocks_[i].NumXIncrements();
  }


  int 
  RdBlkInfo::SubBlockStartingSite (int i) const
  {
    return sub_blocks_[i].StartingSite();
  }

  int
  RdBlkInfo::NodeNumber (void) const
  {
    return rank_;
  }

  void
  RdBlkInfo::NodeNumber (int n)
  {
    rank_ = n;
  }
  
  ostream&
  operator << (ostream& out, const RdBlkInfo& info)
  {
    out << "rank : " << info.rank_ << " tsize: " << info.total_size_ << " ";
    out << "xinc size : " << info.xinc_tot_size_ << " ";
    out << "xinc number of sites : " << info.xinc_sites_ << " ";
    out << "max subblock size : " <<  info.max_sub_blksize_ << " ";
    out << "number of sub blocks " << info.sub_blocks_.size() << " ";
    for (int i = 0; i < info.sub_blocks_.size(); i++) {
      out << "sub block size : " << info.SubBlockSize(i) << " ";
      out << "sub block xinc : " << info.SubBlockXIncrements(i) << " ";
      out << "starting site  : " << info.SubBlockStartingSite(i) << " ";
    }
    return out;
  }
}

