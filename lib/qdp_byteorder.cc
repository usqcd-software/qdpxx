//
//
//  Determine the byte order of a platform
//
//   returns
//     1  = big-endian    (alpha, intel linux, etc.)
//     2  = little-endian (sun, ibm, hp, etc)
//

#include <iostream>
#include <cstdlib>
#include "qdp_byteorder.h"

namespace QDPUtil
{
  //! Is the native byte order big endian?
  bool big_endian()
  {
    int i = 1;
    return (((char*)&i)[sizeof(int) - 1] == 1);
  }

  //! Byte-swap an array of data each of size nmemb
  // NOTE: p has no special alignment requirement
  template <size_t size>
  static void byte_swap(char* p, size_t nmemb)
  {
    // Quick exit
    if (size <= 1)
      return;

    // Change the endianness
    for (size_t i = 0; i < nmemb; i++)
      for (size_t j = 0; j < size / 2; j++)
	std::swap(p[i * size + j], p[i * size + size - 1 - j]);
  }

  void byte_swap(void* ptr, size_t size, size_t nmemb)
  {
    if (size == 1)
      byte_swap<1>((char*)ptr, nmemb);
    else if (size == 2)
      byte_swap<2>((char*)ptr, nmemb);
    else if (size == 4)
      byte_swap<4>((char*)ptr, nmemb);
    else if (size == 8)
      byte_swap<8>((char*)ptr, nmemb);
    else if (size == 16)
      byte_swap<16>((char*)ptr, nmemb);
    else
    {
      std::cerr << __func__ << ": unsupported word size = " << size << "\n";
      exit(1);
    }
  }

  //! fread on a binary file written in big-endian order
  size_t bfread(void *ptr, size_t size, size_t nmemb, FILE *stream)
  {
    size_t n;

    n = fread(ptr, size, nmemb, stream);

    if (! big_endian())
    {
      /* little-endian */
      /* Swap */
      byte_swap(ptr, size, n);
    }

    return n;
  }


  //! fwrite to a binary file in big-endian order
  size_t bfwrite(void *ptr, size_t size, size_t nmemb, FILE *stream)
  {
    size_t n;

    if (big_endian())
    {
      /* big-endian */
      /* Write */
      n = fwrite(ptr, size, nmemb, stream);
    }
    else
    {
      /* little-endian */
      /* Swap and write and swap */
      byte_swap(ptr, size, nmemb);
      n = fwrite(ptr, size, nmemb, stream);
      byte_swap(ptr, size, nmemb);
    }

    return n;
  }

} // namespace QDPUtil
