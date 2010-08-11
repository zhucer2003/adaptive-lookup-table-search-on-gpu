#cmake_minimum_required(VERSION 2.8)

option(WITH_WARNINGS "Compile with warnings" ON)
option(BUILD_SHARED_LIBS "Build shared libraries." OFF)

project(OpenCL_EX CXX)

# Set the build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# The OpenCL_EX version number.
set(OpenCL_EX_VERSION_MAJOR 0)
set(OpenCL_EX_VERSION_MINOR 0)
set(OpenCL_EX_VERSION_PATCH 0)
set(OpenCL_EX_VERSION "${OpenCL_EX_VERSION_MAJOR}.${OpenCL_EX_VERSION_MINOR}")
set(OpenCL_EX_VERSION_FULL "${OpenCL_EX_VERSION}.${OpenCL_EX_VERSION_PATCH}")

# Directory where customize cmake files reside
set(CMAKE_MODULE_PATH
  ${OpenCL_EX_SOURCE_DIR}/CMake/Modules ${CMAKE_MODULE_PATH}
)

find_package(OpenCL)

if(OPENCL_FOUND)
  set(HAVE_OPENCL ON)
  include_directories(${OPENCL_INCLUDE_DIRS})
endif(OPENCL_FOUND)

include_directories(${OpenCL_EX_SOURCE_DIR}/src)
include_directories(${OpenCL_EX_BINARY_DIR}/src)

if(WITH_WARNINGS)
  # enable compiler warning flags
  if("${CMAKE_C_COMPILER_ID}" MATCHES "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -pedantic -Wall -Wextra -Wundef -Wshadow -Wno-unused-parameter")
  endif("${CMAKE_C_COMPILER_ID}" MATCHES "GNU")
  if("${CMAKE_C_COMPILER_ID}" MATCHES "Intel")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -w2 -Wcheck")
  endif("${CMAKE_C_COMPILER_ID}" MATCHES "Intel")
  if ("${CMAKE_C_COMPILER_ID}" MATCHES "PGI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Minform=inform")
  endif("${CMAKE_C_COMPILER_ID}" MATCHES "PGI")
endif(WITH_WARNINGS)

subdirs(src)
