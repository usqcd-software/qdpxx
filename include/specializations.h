// -*- C++ -*-
// $Id: specializations.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);


template<>
struct CreateLeaf<OScalar<IntReal> >
{
  typedef OScalar<IntReal> Leaf_t;
  inline static
  Leaf_t make(const OScalar<IntReal> &a) { return Leaf_t(a); }
};


template<>
struct CreateLeaf<OScalar<IntDouble> >
{
  typedef OScalar<IntDouble> Leaf_t;
  inline static
  Leaf_t make(const OScalar<IntDouble> &a) { return Leaf_t(a); }
};


template<>
struct CreateLeaf<OScalar<IntInteger> >
{
  typedef OScalar<IntInteger> Leaf_t;
  inline static
  Leaf_t make(const OScalar<IntInteger> &a) { return Leaf_t(a); }
};


template<>
struct CreateLeaf<OScalar<IntBoolean> >
{
  typedef OScalar<IntBoolean> Leaf_t;
  inline static
  Leaf_t make(const OScalar<IntBoolean> &a) { return Leaf_t(a); }
};




QDP_END_NAMESPACE();

