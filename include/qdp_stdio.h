// -*- C++ -*-
// $Id: qdp_stdio.h,v 1.1 2003-09-25 22:13:28 edwards Exp $

/*! @file
 * @brief Parallel version of stdio
 *
 * These are the QDP parallel versions of the STL cout,cerr,cin.
 * The user can control which nodes are involved in output/input.
 */

#include <istream>
#include <ostream>
#include <streambuf>
#include <string>

QDP_BEGIN_NAMESPACE(QDP);


/*! @defgroup stdio STD IO
 *
 * Standard-IO-like input and output operations on QDP types
 *
 * @{
 */

//--------------------------------------------------------------------------------
//! StandardInputStream class
/*! Parallel version of standard input
 */
class StandardInputStream
{
public:
  //! Constructor
  StandardInputStream();

  //! Constructor from a streambuf
  StandardInputStream(std::streambuf* b);

  //! destructor
  ~StandardInputStream();

  //! Redirect input stream
  void rdbuf(std::streambuf* b);

  //! Return true if some failure occurred in previous IO operation
  bool fail();

  // Overloaded input functions
  StandardInputStream& operator>>(std::string& input);
  StandardInputStream& operator>>(char& input);
  StandardInputStream& operator>>(int& input);
  StandardInputStream& operator>>(unsigned int& input);
  StandardInputStream& operator>>(short int& input);
  StandardInputStream& operator>>(unsigned short int& input);
  StandardInputStream& operator>>(long int& input);
  StandardInputStream& operator>>(unsigned long int& input);
  StandardInputStream& operator>>(float& input);
  StandardInputStream& operator>>(double& input);
  StandardInputStream& operator>>(bool& input);


private:
  //! Hide copy constructor and =
//  StandardInputStream(const StandardInputStream&) {}
  void operator=(const StandardInputStream&) {}


protected:
  // The universal data-reader. All the read functions call this
  template<typename T>
  StandardInputStream& readPrimitive(T& output);

  // Get the internal ostream
  std::istream& getIstream() {return is;}

private:
  std::istream is;
};




//--------------------------------------------------------------------------------
//! StandardOutputStream class
/*! Parallel version of standard output
 */
class StandardOutputStream
{
public:
  //! Constructor
  StandardOutputStream();

  //! Constructor from a streambuf
  StandardOutputStream(std::streambuf* b);

  //! destructor
  ~StandardOutputStream();

  //! Redirect output stream
  void rdbuf(std::streambuf* b);

  //! Flush the buffer
  void flush();

  //! Return true if some failure occurred in previous IO operation
  bool fail();

  // Overloaded output functions
  StandardOutputStream& operator<<(std::string& output);
  StandardOutputStream& operator<<(char* output);
  StandardOutputStream& operator<<(char output);
  StandardOutputStream& operator<<(int output);
  StandardOutputStream& operator<<(unsigned int output);
  StandardOutputStream& operator<<(short int output);
  StandardOutputStream& operator<<(unsigned short int output);
  StandardOutputStream& operator<<(long int output);
  StandardOutputStream& operator<<(unsigned long int output);
  StandardOutputStream& operator<<(float output);
  StandardOutputStream& operator<<(double output);
  StandardOutputStream& operator<<(bool output);


private:
  //! Hide copy constructor and =
//  StandardOutputStream(const StandardOutputStream&) {}
  void operator=(const StandardOutputStream&) {}


protected:
  // The universal data-writer. All the write functions call this
  template<typename T>
  StandardOutputStream& writePrimitive(T output);

  // Get the internal ostream
  std::ostream& getOstream() {return os;}

private:
  std::ostream os;
};


/*! @} */   // end of group stdio

} // namespace QDPUtil
