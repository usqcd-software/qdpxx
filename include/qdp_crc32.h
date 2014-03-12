/* Code source 
   http://www.ovmj.org/GNUnet/doxygen/html/crc32_8h-source.html
*/
/*
     This file is part of GNUnet

     GNUnet is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published
     by the Free Software Foundation; either version 2, or (at your
     option) any later version.

     GNUnet is distributed in the hope that it will be useful, but
     WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
     General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with GNUnet; see the file COPYING.  If not, write to the
     Free Software Foundation, Inc., 59 Temple Place - Suite 330,
     Boston, MA 02111-1307, USA.
*/

#ifndef __qdp_crc32_h__
#define __qdp_crc32_h__

// #include <qio_stdint.h>

typedef uint32_t uLong;            /* At least 32 bits */

#define Z_NULL  0  

uLong crc32(uLong crc, char const * buf, size_t len);

uLong crc32_4bytes_little_endian (const void* data, size_t length, uint32_t previousCrc32);
uLong crc32_4bytes_big_endian (const void* data, size_t length, uint32_t previousCrc32);
uLong crc32_8bytes_little_endian (const void* data, size_t length, uint32_t previousCrc32);
uLong crc32_8bytes_big_endian (const void* data, size_t length, uint32_t previousCrc32);

#endif
