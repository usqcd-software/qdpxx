// -*- C++ -*-

#ifndef QDP_FLOPCOUNT_H
#define QDP_FLOPCOUNT_H

QDP_BEGIN_NAMESPACE(QDP);

  /*! @defgroup Flop Counting Mechanism
   *
   * This is a basic flop counter class. It is cleared on instantiation
   * but can be cleared at any time by the user. It has functions
   * to add various kinds of flopcount and to retreive the total
   * accumulated flopcount.
   * @{
   */

  //---------------------------------------------------------------------
  //! Basic Flop Counter Clas
  class FlopCounter {
  public:
    //! Constructor - zeroes flopcount
    FlopCounter(void) {
      count = 0;
    }

    //! Destructor - kills object. No cleanup needed
    ~FlopCounter() {} 

    //! Copy Constructor
    FlopCounter(const FlopCounter& c) : count(c.count) {}

    //! Explicit zero method. Clears flopcounts
    inline void reset(void) { 
      count = 0;
    }

    //! Method to add raw number of flops (eg from Level 3 operators)
    inline void addFlops(unsigned long flops) { 
      count += flops;
    }

    //! Method to add per site flop count. Count is multiplied by sitesOnNode()
    inline void addSiteFlops(unsigned long flops) { 
      count += (flops * (unsigned long)Layout::sitesOnNode());
    }

    //! Method to add per site flop count for a subset of sites. Count is multiplied by the site table size of the subset (ie number of sites in a subset)
    inline void addSiteFlops(unsigned long flops, const Subset& s) {
      count += (flops * (unsigned long)(s.numSiteTable()));
    }

    //! Method to retrieve accumulated flopcount
    inline const unsigned long getFlops(void) const { 
      return count;
    }

    //! Report floppage
    inline const void report(const std::string& name, 
			     const Real& time_in_seconds) {

      Real mflops_per_cpu = Real(count)/(Real(1000*1000)*time_in_seconds);
      Real mflops_overall = mflops_per_cpu;
      Internal::globalSum(mflops_overall);
      Real gflops_overall = mflops_overall/Real(1000);
      Real tflops_overall = gflops_overall/Real(1000);

      QDPIO::cout <<"QDP:FlopCount:" << name << " Performance/CPU: t=" << time_in_seconds << "(s) Flops=" << count << " => " << mflops_per_cpu << " Mflops/cpu." << endl;
      QDPIO::cout << "QDP:FlopCount:"  << name <<" Total performance:  " << mflops_overall << " Mflops = " << gflops_overall << " Gflops = " << tflops_overall << " Tflops" << endl;
    }

  private:
    unsigned long count;
  };

  /*! @} */  // end of group 

QDP_END_NAMESPACE();

#endif
