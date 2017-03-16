cmake_minimum_required(VERSION 2.8)

project (poisson3d)

# Variables to be passed to cmake:
# CMAKE_BUILD_TYPE ('Debug' or 'Release')
# PETSC_DIR (directory)
# PETSC_ARCH (string)
# USE_OMP (bool)

# to compile with OpenMP
if(USE_OMP)
	message(STATUS "Compiling with OpenMP")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
		message(STATUS "GNU compiler found.")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
	else()
		message(SEND_ERROR "Don't know OpenMP flag for this compiler")
	endif()
endif()

# set compile options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	message(STATUS "GNU or Clang compiler found.")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ftree-vectorizer-verbose=2")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# PETSc
find_library(PETSC_LIB NAMES petsc PATHS $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
include_directories($ENV{PETSC_DIR}/include $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)

# Hipersolver
find_library(HIPERSOLVER_LIB NAMES hipersolver PATHS ${CMAKE_SOURCE_DIR}/../build)
if("${HIPERSOLVER_LIB}" STREQUAL "HIPERSOLVER_LIB-NOTFOUND")
    message(STATUS "Hipersolver library not found!")
else()
   message(STATUS "Hipersolver library found. Setting corresponding macro variable.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_HIPERSOLVER")
endif()
include_directories(${CMAKE_SOURCE_DIR}/../Hipersolver)

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
