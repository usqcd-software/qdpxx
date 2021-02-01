cmake_minimum_required(VERSION 3.15)
project(QDPXX VERSION 1.46.1 LANGUAGES CXX;C)

include(GNUInstallDirs)
include(FindOpenMP)

include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)

option(QDP_ARCH_PARSCALARVEC "Enable Parscalarvec Arch" OFF)
option(QDP_ARCH_PARSCALAR    "Enable Parscalar Arch default OFF" OFF)
option(QDP_ARCH_SCALAR       "Enable Scalar Arch default ON" ON)
option(QDP_ARCH_SCALARVEC	 "Enable Scalarvec Arch default OFF" OFF)
set(QDP_ND 4 CACHE STRING "Number of Spacetime Dimension (default 4)")
set(QDP_NC 3 CACHE STRING "Number of colors (default 3)")
set(QDP_NS 4 CACHE STRING "Number of spins (default 4)")
option(QDP_LAYOUT_LEXICO "Enable Lexicographic Layout (default OFF)" OFF)
option(QDP_LAYOUT_CB2 "Enable CB2 (4D checkerboarded) layout (default ON)" ON)
option(QDP_LAYOUT_CB32 "Enable CB32 (32-colored) layout (default OFF)" OFF)
option(QDP_LAYOUT_CB3D "Enable CB3D (3D checkerboarded) layout (default OFF)" OFF)
option(QDP_USE_OPENMP  "Enable use of OpenMP" ON)
option(QDP_PRECISION_FLOAT "Use single/float as base precision (otherwise double)" OFF)
set(QDP_ALIGNMENT_SIZE "64" CACHE STRING "OLattice memory alignment size in bytes (default 512)")
option(QDP_USE_GENERIC_OPTS "Enable Generic C specializations" OFF)
option(QDP_USE_PROFILING "Enable Profiling interface" OFF)
option(QDP_DEBUG_MEMORY "Enable Memory Debugging" OFF)
option(QDP_USE_LIBXML2 "Use Libxml2 library" ON)

option(QDP_USE_FILEDB  "Enable FileDB Usage" ON)
option(QDP_USE_SSE  "Enable SSE Code" OFF)
option(QDP_USE_SSE2 "Enable SSE2 Code" OFF)
option(QDP_USE_SSE3 "Enable SSE3 Code" OFF)

option(QDP_USE_HDF5 "Enable HDF5 Code" OFF)

# Deprecate 3DNow option
# Deprecate BG/L option
# Deprecate BG/Q Thread binding option

option(QDP_USE_TBBPOOL_ALLOCATOR "Enable TBB based Pool Allocator" OFF)
option(QDP_EXTRA_MESSAGES "Enable Extra diagnostic messages during compile time" OFF)


mark_as_advanced(QDP_NC, QDP_ND, QDP_NS, QDP_ALIGNMENT_SIZE, QDP_USE_PROFILING, QDP_DEBUG_MEMORY, QDP_USE_HDF5)

message(STATUS "QDP++:  Configuring System")
message(STATUS "QDP++:  Nc=${QDP_NC}")
message(STATUS "QDP++:  Nd=${QDP_ND}")
message(STATUS "QDP++:  Ns=${QDP_NS}")


if ( QDP_ARCH_PARSCALAR ) 
	message(STATUS "QDP++: Configuring for Parscalar build" )
	
	#Check other arches not switched on
	if( QDP_ARCH_PARSCALARVEC OR QDP_ARCH_SCARLARVEC OR QDP_ARCH_SCALAR )
	   message(ERROR "Only one QDP_ARCH_XXX Option should be on at any one time.")
	endif()
	
	# Turn on Parallel QIO
	set(QIO_ENABLE_PARALLEL_BUILD ON CACHE BOOL "Parscalar build: force QIO Parallel build" 
		FORCE)
		
	# Set variable in qdp_config_internal.h -- Autoconf backward compatibility
	set(ARCH_PARSCALAR ON)
elseif( QDP_ARCH_PARSCALARVEC )
	message(STATUS "QDP++: Configuring for Parscalarvec build" ) 
	
	#Check other arches not switched on
	if( QDP_ARCH_PARSCALAR OR QDP_ARCH_SCARLARVEC OR QDP_ARCH_SCALAR )
	   message(ERROR "Only one QDP_ARCH_XXX Option should be on at any one time.")
	endif()
	
	# Set QIO Build 
	set(QIO_ENABLE_PARALLEL_BUILD ON CACHE BOOL "Parscalarvec build: force QIO Parallel build"
		FORCE)
		
	# set option in qdp_config_internal.h
	set(ARCH_PARSCALARVEC ON)
	
elseif( QDP_ARCH_SCALARVEC )
    message(STATUS "QDP++: Configuring for Scalar build" ) 
    
    # Check other archs
    if( QDP_ARCH_PARSCALARVEC OR QDP_ARCH_SCARLAR OR QDP_ARCH_PARSCALAR )
	   message(ERROR "Only one QDP_ARCH_XXX Option should be on at any one time.")
	endif()
    
    # Set QIO Parallel build to false
	set(QIO_ENABLE_PARALLEL_BUILD OFF CACHE BOOL "Scalar buld: Force QIO Scalar build"
        FORCE )
        
    # Set ARCH_SCALAR for qdp_config_internal.h     
    set(ARCH_SCALARVEC ON)
    	
elseif( QDP_ARCH_SCALAR )
    message(STATUS "QDP++: Configuring for Scalar build" ) 
    
    # Check other archs
    if( QDP_ARCH_PARSCALARVEC OR QDP_ARCH_SCARLARVEC OR QDP_ARCH_PARSCALAR )
	   message(ERROR "Only one QDP_ARCH_XXX Option should be on at any one time.")
	endif()
    
    # Set QIO Parallel build to false
	set(QIO_ENABLE_PARALLEL_BUILD OFF CACHE BOOL "Scalar buld: Force QIO Scalar build"
        FORCE )
        
    # Set ARCH_SCALAR for qdp_config_internal.h     
    set(ARCH_SCALAR ON)   
else()
    message(ERROR "QDP++: Unknown architecture. Please set QDPXX_ENABLE_PARSCALAR_ARCH, QDPXX_ENABLE_PARSCALARVEC_ARCH, or QDPXX_ENABLE_SCALAR_ARCH")
endif()


if( QDP_LAYOUT_LEXICO )
  message( STATUS "QDP: Enabling Lexicographic Layout" )
  
  # Check other layouts
  if( QDP_LAYOUT_CB2 OR QDP_LAYOUT_CB32 OR QDP_LAYOUT_CB3D ) 
     message( ERROR "QDP: Can only enable one QDP_LAYOUT_XXX at a time" )
  endif()
  
  set(QDP_USE_LEXICO_LAYOUT ON)
  
elseif( QDP_LAYOUT_CB2 )
  message( STATUS "QDP: Enabling CB2 (4D Checkerboard) Layout" )
  
  # Check other layouts
  if( QDP_LAYOUT_LEXICO OR QDP_LAYOUT_CB32 OR QDP_LAYOUT_CB3D ) 
     message( ERROR "QDP: Can only enable one QDP_LAYOUT_XXX at a time" )
  endif()
  
  set(QDP_USE_CB2_LAYOUT ON)
  
elseif( QDP_LAYOUT_CB32 )
  message( STATUS "QDP: Enabling CB32 (32 colored ) Layout" )
  
  # Check other layouts
  if( QDP_LAYOUT_LEXICO OR QDP_LAYOUT_CB2 OR QDP_LAYOUT_CB3D ) 
     message( ERROR "QDP: Can only enable one QDP_LAYOUT_XXX at a time" )
  endif()
  
  set(QDP_USE_CB32_LAYOUT ON)

elseif( QDP_LAYOUT_CB3D )
  message( STATUS "QDP: Enabling CB3D (3D checkerboarded ) Layout" )
  
  # Check other layouts
  if( QDP_LAYOUT_LEXICO OR QDP_LAYOUT_CB2 OR QDP_LAYOUT_CB32 ) 
     message( ERROR "QDP: Can only enable one QDP_LAYOUT_XXX at a time" )
  endif()
  
  set(QDP_USE_CB3D_LAYOUT ON)
else()
  message( ERROR "QDP: Unknown layout. Please set either QDP_LAYOUT_LEXICO, QDP_LAYOUT_CB2, QDP_LAYOUT_CB3D or QDP_LAYOUT_CB32")
endif()

if( QDP_ENABLE_EXTRA_MESSAGES )
  set(QPHIX_EMIT_MESSAGES ON) # FIXME: Refactor this later to use QDP
  message( STATUS "Enabling pragma messages" )
endif()	

if( QDP_PRECISION_FLOAT )
  set(BASE_PRECISION 32)
else()
  set(BASE_PRECISION 64)
endif()

# Autoconf variable name backard compatibility 
set(QDP_AC_ALIGNMENT_SIZE ${QDP_ALIGNMENT_SIZE})
  


if ( QDP_ARCH_PARSCALAR OR QDP_ARCH_PARSCALARVEC )
  find_package(QMP REQUIRED)
endif()

if ( QDP_USE_HDF5 ) 
  find_package(HDF5 REQUIRED)
endif()

if ( QDP_USE_FILEDB ) 
   set(BUILD_FILEDB ON)
endif()

if ( QDP_USE_OPENMP )
   set(QDP_USE_OMP_THREADS ON)
   find_package(OpenMP REQUIRED)
endif()

if( QDP_USE_TBBPOOL_ALLOCATOR ) 
   find_package(TBB REQUIRED )
endif()
configure_file(include/qdp_config_internal.h.cmake.in include/qdp_config_internal.h)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/qdp_config_internal.h DESTINATION include)
add_subdirectory(include)


if (QDP_USE_LIBXML2 ) 
  add_subdirectory(other_libs/xpath_reader)
endif()

if (QDP_USE_FILEDB )
  add_subdirectory(other_libs/filedb)
endif()

if( QDP_USE_SSE2 ) 
  add_subdirectory(other_libs/libintrin)
endif()

add_subdirectory(other_libs/qio)
