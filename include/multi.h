// -*- C++ -*-
// $Id: multi.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// Support for reference semantic multi-dimensional arrays
//

#ifndef MULTI_INCLUDE
#define MULTI_INCLUDE

QDP_BEGIN_NAMESPACE(QDP);

  //! Container for a multi-dimensional 1D array
template<class T> class multi1d
{
public:
  multi1d() {F=0;copymem=false;}
  multi1d(T *f, int ns1) {F=f; n1=ns1;copymem=true;}
  explicit multi1d(int ns1) {copymem=false;F=0;resize(ns1);}
  ~multi1d() {if (! copymem) {delete[] F;}}

  //! Copy constructor
  multi1d(const multi1d& s): copymem(false), n1(s.n1), F(0)
    {
      resize(n1);

      for(int i=0; i < n1; ++i)
	F[i] = s.F[i];
    }

  //! Allocate mem for the array
  void resize(int ns1) 
    {if(copymem) {cerr<<"invalid resize in 1d\n";exit(1);}; delete[] F; n1=ns1; F = new T[n1];}

  //! Size of array
  const int size() const {return n1;}
  const int size1() const {return n1;}

  //! Equal operator uses underlying = of T
  multi1d<T>& operator=(const multi1d<T>& s1)
    {
      if (F == 0)
	resize(s1.size());

      for(int i=0; i < n1; ++i)
	F[i] = s1.F[i];
      return *this;
    }

  //! Equal operator uses underlying = of T
  template<class T1>
  multi1d<T>& operator=(const T1& s1)
    {
      if (F == 0)
      {
	cerr << "left hand side not initialized\n";
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] = s1;
      return *this;
    }

  //! Set equal to a old-style C 1-D array
  multi1d<T>& operator=(const T s1[])
    {
      if (F == 0)
      {
	cerr << "left hand side not initialized\n";
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] = s1[i];
      return *this;
    }

  //! Return ref to a column slice
  const T* slice() const {return F;}

  //! Return ref to an element
  T& operator()(int i) {return F[i];}

  //! Return const ref to an element
  const T& operator()(int i) const {return F[i];}

  //! Return ref to an element
  T& operator[](int i) {return F[i];}

  //! Return const ref to an element
  const T& operator[](int i) const {return F[i];}

private:
  bool copymem;
  int n1;
  T *F;
};

//! Container for a multi-dimensional 2D array
template<class T> class multi2d
{
public:
  multi2d() {F=0;n1=n2=sz=0;copymem=false;}
  multi2d(T *f, int ns2, int ns1) {F=f; n1=ns1; n2=ns2; sz=n1*n2; copymem=true;}
  explicit multi2d(int ns2, int ns1) {copymem=false;F=0;resize(ns2,ns1);}
  ~multi2d() {if (! copymem) {delete[] F;}}

  //! Copy constructor
  multi2d(const multi2d& s): copymem(false), n1(s.n1), n2(s.n2), sz(s.sz), F(0)
    {
      resize(n2,n1);

      for(int i=0; i < sz; ++i)
	F[i] = s.F[i];
    }

  //! Allocate mem for the array
  void resize(int ns2, int ns1) 
    {if(copymem) {cerr<<"invalid resize in 2d\n";exit(1);}; delete[] F; 
    n1=ns1; n2=ns2; sz=n1*n2; F = new T[sz];}

  //! Size of array
  const int size1() const {return n1;}
  const int size2() const {return n2;}

  //! Equal operator uses underlying = of T
  multi2d<T>& operator=(const multi2d<T>& s1)
    {
      if (F == 0)
	resize(s1.size2(), s1.size1());

      for(int i=0; i < sz; ++i)
	F[i] = s1.F[i];
      return *this;
    }

  //! Equal operator uses underlying = of T
  template<class T1>
  multi2d<T>& operator=(const T1& s1)
    {
      if (F == 0)
      {
	cerr << "left hand side not initialized\n";
	exit(1);
      }

      for(int i=0; i < sz; ++i)
	F[i] = s1;
      return *this;
    }

  //! Return ref to a row slice
  const T* slice(int j) const {return F+n1*j;}

  //! Return ref to an element
  T& operator()(int j, int i) {return F[i+n1*j];}

  //! Return const ref to an element
  const T& operator()(int j, int i) const {return F[i+n1*j];}

  //! Return ref to an element
  multi1d<T> operator[](int j) {return multi1d<T>(F+j*n1,n1);}

  //! Return const ref to an element
  const multi1d<T> operator[](int j) const {return multi1d<T>(F+j*n1,n1);}

private:
  bool copymem;
  int n1;
  int n2;
  int sz;
  T *F;
};

//! Container for a multi-dimensional 3D array
template<class T> class multi3d
{
public:
  multi3d() {F=0;n1=n2=n3=sz=0;copymem=false;}
  multi3d(T *f, int ns3, int ns2, int ns1) {F=f; n1=ns1; n2=ns2; n3=ns3; sz=n1*n2*n3; copymem=true;}
  explicit multi3d(int ns3, int ns2, int ns1) {copymem=false;F=0;resize(ns3,ns2,ns1);}
  ~multi3d() {if (! copymem) {delete[] F;}}

  //! Copy constructor
  multi3d(const multi3d& s): copymem(false), n1(s.n1), n2(s.n2), n3(s.n3), sz(s.sz), F(0)
    {
      resize(n3,n2,n1);

      for(int i=0; i < sz; ++i)
	F[i] = s.F[i];
    }

  //! Allocate mem for the array
  void resize(int ns3, int ns2, int ns1) 
    {if(copymem) {cerr<<"invalid resize in 2d\n";exit(1);}; delete[] F; 
    n1=ns1; n2=ns2; n3=ns3; sz=n1*n2*n3; F = new T[sz];}

  //! Size of array
  const int size1() const {return n1;}
  const int size2() const {return n2;}
  const int size3() const {return n3;}

  //! Equal operator uses underlying = of T
  multi3d<T>& operator=(const multi3d<T>& s1)
    {
      if (F == 0)
	resize(s1.size3(), s1.size2(), s1.size());

      for(int i=0; i < sz; ++i)
	F[i] = s1.F[i];
      return *this;
    }

  //! Equal operator uses underlying = of T
  template<class T1>
  multi3d<T>& operator=(const T1& s1)
    {
      if (F == 0)
      {
	cerr << "left hand side not initialized\n";
	exit(1);
      }

      for(int i=0; i < sz; ++i)
	F[i] = s1;
      return *this;
    }

  //! Return ref to a column slice
  const T* slice(int k, int j) const {return F+n1*(j+n2*(k));}

  //! Return ref to an element
  T& operator()(int k, int j, int i) {return F[i+n1*(j+n2*(k))];}

  //! Return const ref to an element
  const T& operator()(int k, int j, int i) const {return F[i+n1*(j+n2*(k))];}

  //! Return ref to an element
  multi2d<T> operator[](int k) {return multi2d<T>(F+n1*n2*k,n2,n1);}

  //! Return const ref to an element
  const multi2d<T> operator[](int k) const {return multi2d<T>(F+n1*n2*k,n2,n1);}

private:
  bool copymem;
  int n1;
  int n2;
  int n3;
  int sz;
  T *F;
};


//! Container for a multi-dimensional 4D array
template<class T> class multi4d
{
public:
  multi4d() {F=0;n1=n2=n3=sz=0;}
  explicit multi4d(int ns4, int ns3, int ns2, int ns1) {F=0;resize(ns4,ns3,ns2,ns1);}
  ~multi4d() {delete[] F;}

  //! Copy constructor
  multi4d(const multi4d& s): n1(s.n1), n2(s.n2), n3(s.n3), n4(s.n4), sz(s.sz), F(0)
    {
      resize(n4,n3,n2,n1);

      for(int i=0; i < sz; ++i)
	F[i] = s.F[i];
    }

  //! Allocate mem for the array
  void resize(int ns4, int ns3, int ns2, int ns1) 
    {delete[] F; n1=ns1; n2=ns2; n3=ns3; n4=ns4; sz=n1*n2*n3*n4; F = new T[sz];}

  //! Size of array
  const int size1() const {return n1;}
  const int size2() const {return n2;}
  const int size3() const {return n3;}
  const int size4() const {return n4;}

  //! Equal operator uses underlying = of T
  multi4d<T>& operator=(const multi4d<T>& s1)
    {
      if (F == 0)
	resize(s1.size4(), s1.size3(), s1.size2(), s1.size());

      for(int i=0; i < sz; ++i)
	F[i] = s1.F[i];
      return *this;
    }

  //! Equal operator uses underlying = of T
  template<class T1>
  multi4d<T>& operator=(const T1& s1)
    {
      if (F == 0)
      {
	cerr << "left hand side not initialized\n";
	exit(1);
      }

      for(int i=0; i < sz; ++i)
	F[i] = s1;
      return *this;
    }

  //! Return ref to a column slice
  const T* slice(int l, int k, int j) const {return F+n1*(j+n2*(k+n3*(l)));}

  //! Return ref to an element
  T& operator()(int l, int k, int j, int i) {return F[i+n1*(j+n2*(k+n3*(l)))];}

  //! Return const ref to an element
  const T& operator()(int l, int k, int j, int i) const {return F[i+n1*(j+n2*(k+n3*(l)))];}

  //! Return ref to an element
  multi3d<T> operator[](int l) {return multi3d<T>(F+n1*n2*n3*l,n3,n2,n1);}

  //! Return const ref to an element
  const multi3d<T> operator[](int l) const {return multi3d<T>(F+n1*n2*n3*k,n3,n2,n1);}

private:
  int n1;
  int n2;
  int n3;
  int n4;
  int sz;
  T *F;
};

QDP_END_NAMESPACE();

#endif
