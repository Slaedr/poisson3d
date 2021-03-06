cmake_minimum_required(VERSION 2.8)

project (poisson3d)

# Variables to be passed to cmake:
# CMAKE_BUILD_TYPE ('Debug' or 'Release')
# OMP: non-zero for compiling with OpenMP
# BUILD_KNC: non-zero to build for Xeon Phi KNC

# Environment variables needed:
# PETSC_DIR (directory)
# PETSC_ARCH (string)
# HIPERSOLVER_DIR location of PHiPerSolver source tree
#  libhipersolver.so is assumed to be in $HIPERSOLVER_DIR/build.

if(BUILD_KNC)
	message(STATUS "Building for KNC, with MKL")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mmic")
	set (CMAKE_EXECUTABLE_SUFFIX ".MIC")
	include_directories($ENV{MKLROOT}/include)
endif()

# to compile with OpenMP
if(OMP)
	message(STATUS "Compiling with OpenMP")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
		message(STATUS "GNU compiler found.")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
	else()
		message(SEND_ERROR "Don't know OpenMP flag for this compiler")
	endif()
endif()

# set compile options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	message(STATUS "GNU or Clang compiler found.")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ftree-vectorizer-verbose=2")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Winline -qopt-report=2")
endif()

# PETSc
find_library(PETSC_LIB NAMES petsc PATHS $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
include_directories($ENV{PETSC_DIR}/include $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)

# Hipersolver
find_library(HIPERSOLVER_LIB NAMES hipersolver PATHS $ENV{HIPERSOLVER_DIR}/build)
if("${HIPERSOLVER_LIB}" STREQUAL "HIPERSOLVER_LIB-NOTFOUND")
    message(STATUS "Hipersolver library not found!")
else()
   message(STATUS "Hipersolver library found.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_HIPERSOLVER")
endif()
include_directories($ENV{HIPERSOLVER_DIR})

message(STATUS "Hipersolver library = ${HIPERSOLVER_LIB}")

# ---------------------------------------------------------------------------- #

# flags and stuff

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -ggdb -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -UDEBUG")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug mode")
else()
	message(STATUS "Release mode")
endif()

add_executable(poisson3d poisson3d.cpp)
target_link_libraries(poisson3d ${PETSC_LIB} ${HIPERSOLVER_LIB})
if(BUILD_KNC)
	target_link_libraries(poisson3d $ENV{MKLROOT}/lib/mic/libmkl_rt.so m dl pthread)
endif()
