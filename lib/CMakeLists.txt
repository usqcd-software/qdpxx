

add_library(qdp qdp_map.cc 
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
   target_sources(qdp PRIVATE
   		qdp_scalar_init.cc 
   		qdp_scalar_layout.cc
		qdp_scalar_specific.cc
		qdp_scalarsite_specific.cc
   )
endif()

if( QDP_ARCH_PARSCALAR ) 
	target_sources(qdp PRIVATE 
		qdp_parscalar_init.cc 
		qdp_parscalar_layout.cc
		qdp_parscalar_specific.cc
		qdp_scalarsite_specific.cc
	)	
endif()

# Optimized code using sse extensions
if( QDP_ARCH_SCALAR OR QDP_ARCH_PARSCALAR )
	if( QDP_USE_SSE )
		target_sources(qdp PRIVATE 
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
	target_sources(qdp PRIVATE	
		qdp_scalarvec_init.cc 
		qdp_scalarvec_layout.cc
		qdp_scalarvec_specific.cc
		qdp_scalarvecsite_specific.cc
	)
endif()

if( QDP_ARCH_PARSCALARVEC )
	target_sources(qdp PRIVATE
		qdp_parscalarvec_init.cc 
		qdp_parscalarvec_layout.cc
		qdp_parscalarvec_specific.cc 
		qdp_scalarvecsite_specific.cc
	)
endif()

if( QDP_ARCH_SCALARVEC OR QDP_ARCH_PARSCALARVEC )
  if( QDP_USE_SSE ) 
  	target_sources(qdp PRIVATE qdp_scalarvecsite_sse.cc )
  endif()
endif()

if( QDP_USE_LIBXML2 )
   target_sources(qdp PRIVATE 
   	     qdp_xmlio.cc 
   	     qdp_iogauge.cc
		 qdp_qdpio.cc
		 qdp_qio_strings.cc
		 qdp_map_obj_disk.cc
   )
endif()

if( QDP_USE_HDF5 )
   target_sources(qdp PRIVATE  qdp_hdf5.cc )
endif()

if( QDP_USE_TBBPOOL_ALLOCATOR )
	target_sources(qdp PRIVATE qdp_pool_allocator.cc )
else()
 	target_sources(qdp PRIVATE qdp_default_allocator.cc )
endif()

# Add the current source and build direcories, but in the
# Install interface it all comes from include
target_include_directories( qdp PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
	$<INSTALL_INTERFACE:include> 
)

# Link QIO from subtree
target_link_libraries( qdp PUBLIC QIO::qio )

# If using libxml link the XPathReader and xmlWriter libraries
if( QDP_USE_LIBXML2 ) 
	target_link_libraries( qdp PUBLIC XPathReader::XPathReader XPathReader::xmlWriter )
endif()

# If we have QDP_USE_SSE2 enabled we can use libintrin
if( QDP_USE_SSE2 )
	target_link_libraries( qdp PUBLIC LibIntrin::libintrin )
endif()

# If we want filedb (always?) link that
if( QDP_USE_FILEDB ) 
	target_link_libraries( qdp PUBLIC FileDB::filedb ) 
endif()

if( QDP_USE_TBBPOOL_ALLOCATOR )
	target_link_libraries (qdp PUBLIC TBB::tbbmalloc )
endif()

if( QDP_USE_HDF5 )
    # FIXME Check this
	target_link_libraries( qdp PUBLIC hdf5::hdf5 )
endif()

if( QDP_USE_OPENMP )
  target_link_libraries( qdp PUBLIC OpenMP::OpenMP_CXX )
endif()

if( QDP_SANITIZER_OPTS )
  target_compile_options(qdp PRIVATE ${QDP_SANITIZER_OPTS})
  target_link_options(qdp PUBLIC ${QDP_SANITIZER_OPTS})
endif()

install(TARGETS qdp EXPORT QDPXXTargets
  ARCHIVE DESTINATION lib
  LIBRARY  DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

#install the export description of the targets
install(EXPORT QDPXXTargets
        FILE QDPXXTargets.cmake
        NAMESPACE QDPXX::
        DESTINATION lib/cmake/QDPXX)

add_library(QDPXX::qdp ALIAS qdp)
