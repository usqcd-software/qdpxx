

set( QDPXX_SRCS qdp_map.cc 
	qdp_subset.cc 
	qdp_random.cc
	qdp_layout.cc 
	qdp_io.cc
	qdp_byteorder.cc
	qdp_util.cc
	qdp_stdio.cc
  	qdp_profile.cc 
  	qdp_strnlen.cc 
  	qdp_crc32.cc
  	qdp_stopwatch.cc
	qdp_rannyu.cc
	qdp_generic_threadbind.cc
)


# Scalar	
if( QDP_ARCH_SCALAR )
   set( QDPXX_SRCS ${QDPXX_SRCS}
   		qdp_scalar_init.cc 
   		qdp_scalar_layout.cc
		qdp_scalar_specific.cc
		qdp_scalarsite_specific.cc
   )
endif()

if( QDP_ARCH_PARSCALAR ) 
	set( QDPXX_SRCS ${QDPXX_SRCS}
		qdp_parscalar_init.cc 
		qdp_parscalar_layout.cc
		qdp_parscalar_specific.cc
		qdp_scalarsite_specific.cc
	)	
endif()

# Optimized code using sse extensions
if( QDP_ARCH_SCALAR OR QDP_ARCH_PARSCALAR )
	if( QDP_USE_SSE )
		set(QDPXX_SRCS ${QDPXX_SRCS} 
			scalarsite_sse/qdp_scalarsite_sse.cc 
			scalarsite_sse/qdp_scalarsite_linalg_double.cc 
			scalarsite_sse/sse_blas_local_sumsq_double.cc 
			scalarsite_sse/sse_blas_local_vcdot_double.cc 
			scalarsite_sse/sse_blas_local_vcdot_real_double.cc 
			scalarsite_sse/sse_blas_vaxmbyz4_double.cc 
			scalarsite_sse/sse_blas_vaxmyz4_double.cc 
			scalarsite_sse/sse_blas_vaxpbyz4_double.cc 
			scalarsite_sse/sse_blas_vaxpy4_double.cc 
			scalarsite_sse/sse_blas_vaypx4_double.cc 
			scalarsite_sse/sse_blas_vscal4_double.cc 
			scalarsite_sse/sse_linalg_m_eq_scal_m_double.cc 
			scalarsite_sse/sse_linalg_m_peq_m_double.cc 
			scalarsite_sse/sse_linalg_m_eq_mm_double.cc 
			scalarsite_sse/sse_linalg_m_eq_mh_double.cc 
			scalarsite_sse/sse_linalg_m_eq_hm_double.cc 
			scalarsite_sse/sse_linalg_m_eq_hh_double.cc
		)
	endif()
endif()

if( QDP_ARCH_SCALARVEC )
	set( QDPXX_SRCS ${QDPXX_SRCS} 		
		qdp_scalarvec_init.cc 
		qdp_scalarvec_layout.cc
		qdp_scalarvec_specific.cc
		qdp_scalarvecsite_specific.cc
	)
endif()

if( QDP_ARCH_PARSCALARVEC )
	set( QDPXX_SRCS ${QDPXX_SRCS}
		qdp_parscalarvec_init.cc 
		qdp_parscalarvec_layout.cc
		qdp_parscalarvec_specific.cc 
		qdp_scalarvecsite_specific.cc
	)
endif()

if( QDP_ARCH_SCALARVEC OR QDP_ARCH_PARSCALARVEC )
  if( QDP_USE_SSE ) 
  	set( QDPXX_SRCS ${QDPXX_SRCS} qdp_scalarvecsite_sse.cc )
  endif()
endif()

if( QDP_USE_LIBXML2 )
   set( QDPXX_SRCS ${QDPXX_SRCS}
   	     qdp_xmlio.cc 
   	     qdp_iogauge.cc
		 qdp_qdpio.cc
		 qdp_qio_strings.cc
		 qdp_map_obj_disk.cc
   )
endif()

if( QDP_USE_HDF5 )
   set( QDPXX_SRCS ${QDPXX_SRCS} qdp_hdf5.cc )
endif()

if( BUILD_QDPXX_TBB_ALLOCATOR )
	set( QDPXX_SRCS ${QDPXX_SRCS} qdp_pool_allocator.cc )
else()
 	set( QDPXX_SRCS ${QDPXX_SRCS} qdp_default_allocator.cc )
endif()

# Create the library
add_library(qdp ${QDPXX_SRCS})

# Set includepaths
# These variables just save my fingers
set( OLSRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../other_libs )
set( OLBINDIR ${CMAKE_CURRENT_BINARY_DIR}/../other_libs )

# Add the current source and build direcories, but in the
# Install interface it all comes from include
target_include_directories( qdp PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
	$<INSTALL_INTERFACE:include> 
)

# Link QIO from subtree
target_link_libraries( qdp PUBLIC qio )

# If using libxml link the XPathReader and xmlWriter libraries
if( QDP_USE_LIBXML2 ) 
	target_link_libraries( qdp PUBLIC XPathReader xmlWriter )
endif()

# If we have QDP_USE_SSE2 enabled we can use libintrin
if( QDP_USE_SSE2 )
	target_link_libraries( qdp PUBLIC libintrin )
endif()

# If we want filedb (always?) link that
if( QDP_USE_FILEDB ) 
	target_link_libraries( qdp PUBLIC filedb ) 
endif()

