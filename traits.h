// -*- C++ -*-
// $Id: traits.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);


//-----------------------------------------------------------------------------
// Traits classes to support operations of simple scalars (floating constants, 
// etc.) on QDPTypes
//-----------------------------------------------------------------------------

// Find the underlying word type of a field
template<class T>
struct WordType
{
  typedef T  Type_t;
};


//-----------------------------------------------------------------------------
// Constructors for simple word types
//-----------------------------------------------------------------------------

// Construct simple word type. Default behavior is empty
template<class T>
struct SimpleScalar {};


// Construct simple word type used at some level within primitives
template<class T>
struct InternalScalar {};


// Construct primitive type of input but always RScalar complex type
template<class T>
struct NoComplex {};


QDP_END_NAMESPACE();

