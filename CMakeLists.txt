cmake_minimum_required(VERSION 3.15)
project(QDPXX VERSION 1.46.1 LANGUAGES CXX;C)

include(GNUInstallDirs)

include(FindOpenMP)          # Gives us find_package() for OpenMP

# I have not actually used these -- I dont know if they are needed
#
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)

# Config parameters: ARCHs - one can turn them on/off
option(QDP_ARCH_PARSCALARVEC "Enable Parscalarvec Arch" OFF)
option(QDP_ARCH_PARSCALAR    "Enable Parscalar Arch default OFF" OFF)
option(QDP_ARCH_SCALAR       "Enable Scalar Arch default ON" ON)
option(QDP_ARCH_SCALARVEC	 "Enable Scalarvec Arch default OFF" OFF)

# NC, ND, NS
set(QDP_ND 4 CACHE STRING "Number of Spacetime Dimension (default 4)")
set(QDP_NC 3 CACHE STRING "Number of colors (default 3)")
set(QDP_NS 4 CACHE STRING "Number of spins (default 4)")

# Layouts
option(QDP_LAYOUT_LEXICO "Enable Lexicographic Layout (default OFF)" OFF)
option(QDP_LAYOUT_CB2 "Enable CB2 (4D checkerboarded) layout (default ON)" ON)
option(QDP_LAYOUT_CB32 "Enable CB32 (32-colored) layout (default OFF)" OFF)
option(QDP_LAYOUT_CB3D "Enable CB3D (3D checkerboarded) layout (default OFF)" OFF)

# Precision: Set this to true for pure single prec builds
option(QDP_PRECISION_FLOAT "Use single/float as base precision (otherwise double)" OFF)

# Alignment for OLattice stuff (64=512 bits e.g. KNL, Xeon)
set(QDP_ALIGNMENT_SIZE "64" CACHE STRING "OLattice memory alignment size in bytes (default 512)")

# Generic C Opts
option(QDP_USE_GENERIC_OPTS "Enable Generic C specializations" OFF)

# Profiling and memory debugging
option(QDP_USE_PROFILING "Enable Profiling interface" OFF)
option(QDP_DEBUG_MEMORY "Enable Memory Debugging" OFF)

# LibXML on by default
option(QDP_USE_LIBXML2 "Use Libxml2 library" ON)

# FileDB on by default
option(QDP_USE_FILEDB  "Enable FileDB Usage" ON)

# SSE Levels
option(QDP_USE_SSE  "Enable SSE Code" OFF)
option(QDP_USE_SSE2 "Enable SSE2 Code" OFF)
option(QDP_USE_SSE3 "Enable SSE3 Code" OFF)

# HDF5 from Thorsten
option(QDP_USE_HDF5 "Enable HDF5 Code" OFF)

# Deprecate 3DNow option
# Deprecate BG/L option
# Deprecate BG/Q Thread binding option

# Threaded Building Block (needs TBB installed, and their CMake Location on the Module path
option(QDP_USE_TBBPOOL_ALLOCATOR "Enable TBB based Pool Allocator" OFF)

# From our friends in Bonn - terse compilation
option(QDP_EXTRA_MESSAGES "Enable Extra diagnostic messages during compile time" OFF)

# These options will only apperar in ccmake if you toggle advanced features
mark_as_advanced(QDP_NC, QDP_ND, QDP_NS, QDP_ALIGNMENT_SIZE, QDP_USE_PROFILING, QDP_DEBUG_MEMORY, QDP_USE_HDF5)

# Some status
message(STATUS "QDP++:  Configuring System")
message(STATUS "QDP++:  Nc=${QDP_NC}")
message(STATUS "QDP++:  Nd=${QDP_ND}")
message(STATUS "QDP++:  Ns=${QDP_NS}")


if ( QDP_ARCH_PARSCALAR ) 
	message(STATUS "QDP++: Configuring for Parscalar build" )
	
	#Check other arches not switched on
	if( QDP_ARCH_PARSCALARVEC OR QDP_ARCH_SCARLARVEC OR QDP_ARCH_SCALAR )
	   message(ERROR "QDP++: Only one QDP_ARCH_XXX Option should be on at any one time.")
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
	   message(ERROR "QDP++: Only one QDP_ARCH_XXX Option should be on at any one time.")
	endif()
	
	# Set QIO Build 
	set(QIO_ENABLE_PARALLEL_BUILD ON CACHE BOOL "Parscalarvec build: force QIO Parallel build"
		FORCE)
		
	# set option in qdp_config_internal.h
	set(ARCH_PARSCALARVEC ON)
	
elseif( QDP_ARCH_SCALARVEC )
    message(STATUS "QDP++: Configuring for ScalarVec build" ) 
    
    # Check other archs
    if( QDP_ARCH_PARSCALARVEC OR QDP_ARCH_SCARLAR OR QDP_ARCH_PARSCALAR )
	   message(ERROR "QDP++: Only one QDP_ARCH_XXX Option should be on at any one time.")
	endif()
    
    # Set QIO Parallel build to false
	set(QIO_ENABLE_PARALLEL_BUILD OFF CACHE BOOL "Scalarvec buld: Force QIO Scalar build"
        FORCE )
        
    # Set ARCH_SCALAR for qdp_config_internal.h     
    set(ARCH_SCALARVEC ON)
    	
elseif( QDP_ARCH_SCALAR )
    message(STATUS "QDP++: Configuring for Scalar build" ) 
    
    # Check other archs
    if( QDP_ARCH_PARSCALARVEC OR QDP_ARCH_SCARLARVEC OR QDP_ARCH_PARSCALAR )
	   message(ERROR "QDP++: Only one QDP_ARCH_XXX Option should be on at any one time.")
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
  message( STATUS "QDP++: Enabling Lexicographic Layout" )
  
  # Check other layouts
  if( QDP_LAYOUT_CB2 OR QDP_LAYOUT_CB32 OR QDP_LAYOUT_CB3D ) 
     message( ERROR "QDP++: Can only enable one QDP_LAYOUT_XXX at a time" )
  endif()
  
  set(QDP_USE_LEXICO_LAYOUT ON)
  
elseif( QDP_LAYOUT_CB2 )
  message( STATUS "QDP++: Enabling CB2 (4D Checkerboard) Layout" )
  
  # Check other layouts
  if( QDP_LAYOUT_LEXICO OR QDP_LAYOUT_CB32 OR QDP_LAYOUT_CB3D ) 
     message( ERROR "QDP++: Can only enable one QDP_LAYOUT_XXX at a time" )
  endif()
  
  set(QDP_USE_CB2_LAYOUT ON)
  
elseif( QDP_LAYOUT_CB32 )
  message( STATUS "QDP++: Enabling CB32 (32 colored ) Layout" )
  
  # Check other layouts
  if( QDP_LAYOUT_LEXICO OR QDP_LAYOUT_CB2 OR QDP_LAYOUT_CB3D ) 
     message( ERROR "QDP++: Can only enable one QDP_LAYOUT_XXX at a time" )
  endif()
  
  set(QDP_USE_CB32_LAYOUT ON)

elseif( QDP_LAYOUT_CB3D )
  message( STATUS "QDP++: Enabling CB3D (3D checkerboarded ) Layout" )
  
  # Check other layouts
  if( QDP_LAYOUT_LEXICO OR QDP_LAYOUT_CB2 OR QDP_LAYOUT_CB32 ) 
     message( ERROR "QDP++: Can only enable one QDP_LAYOUT_XXX at a time" )
  endif()
  
  set(QDP_USE_CB3D_LAYOUT ON)
else()
  message( ERROR "QDP++: Unknown layout. Please set either QDP_LAYOUT_LEXICO, QDP_LAYOUT_CB2, QDP_LAYOUT_CB3D or QDP_LAYOUT_CB32")
endif()

if( QDP_ENABLE_EXTRA_MESSAGES )
  set(QPHIX_EMIT_MESSAGES ON) # FIXME: Refactor this later to use QDP
  message( STATUS "QDP++: Enabling pragma messages" )
endif()	

if( QDP_PRECISION_FLOAT )
  set(BASE_PRECISION 32)
else()
  set(BASE_PRECISION 64)
endif()
message( STATUS "QDP++: Setting Base Precision to ${BASE_PRECISION}")

# Autoconf variable name backard compatibility 
set(QDP_AC_ALIGNMENT_SIZE ${QDP_ALIGNMENT_SIZE})
message( STATUS "QDP++: Setting alignment size to ${QDP_ALIGNMENT_SIZE}")
 
# OpenMP Support hardwired now
find_package(OpenMP REQUIRED)
set(QDP_USE_OMP_THREADS "1")

# If we are parallel we need QMP
if ( QDP_ARCH_PARSCALAR OR QDP_ARCH_PARSCALARVEC )
  find_package(QMP REQUIRED)
endif()

# If we use HDF5 we need HDF5
if ( QDP_USE_HDF5 ) 
  find_package(HDF5 REQUIRED)
  message(STATUS "QDP++: Building with HDF5")
endif()

# If we want filedb mark it on 
if ( QDP_USE_FILEDB ) 
   set(BUILD_FILEDB ON)
endif()

# If we are using OpenMP mark appropriately in the configured header
if ( QDP_USE_OPENMP )
   set(QDP_USE_OMP_THREADS ON)
   find_package(OpenMP REQUIRED)
endif()

# If we use Threaded building blocks Find TBB
if( QDP_USE_TBBPOOL_ALLOCATOR ) 
   find_package(TBB REQUIRED )
   message(STATUS "QDP++: Using Threader Building Blocks Pool Allocator")
endif()

# SSE3 => SSE2 (and SSE )
if( QDP_USE_SSE3 ) 
  message(STATUS "QDP++: SSE3 optimizations enabled")
  set(QDP_USE_SSE2 ON)

endif()

# SSE2 => SSE
if( QDP_USE_SSE2 )
  message(STATUS "QDP++: SSE2 optimizations enabled")
  set(QDP_USE_SSE ON)
endif()

if( QDP_USE_SSE )
  message(STATUS "QDP++: SSE optimizations enabled")
endif()

# Write config file with currently set values
configure_file(include/qdp_config_internal.h.cmake.in include/qdp_config_internal.h)

# Install generated file (when make install is called)
# I cannot mark the configured file as a 'target' but I can 
# Use CMAKE_INSTALL_INCLUDEDIR from the GNUInstallDirs to select the standard
# Include dir. On Linux/UNIX it should be CMAKE_INSTALL_PREFIX/include 
#
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/qdp_config_internal.h 
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Configure subdirectories
# Include just does installing
add_subdirectory(include)

# Compile files in lib
add_subdirectory(lib)

# In practice this is always taken, but
# in principle we can make QDP++ not depend on LibXML
# 
if (QDP_USE_LIBXML2 ) 
  add_subdirectory(other_libs/xpath_reader)
endif()

# FileDB nearly always gets built
if (QDP_USE_FILEDB )
  add_subdirectory(other_libs/filedb)
endif()

# If SSE2 is on we can use libintrin
if( QDP_USE_SSE2 )
  add_subdirectory(other_libs/libintrin)
endif()

# We always build QIO
add_subdirectory(other_libs/qio)

# The examples and nersc2ildg etc
add_subdirectory(examples)

# The tests -- test primarily whether SSE implementations 
# are correct so we can do that
if( QDP_USE_SSE )
	add_subdirectory(tests)
endif()
