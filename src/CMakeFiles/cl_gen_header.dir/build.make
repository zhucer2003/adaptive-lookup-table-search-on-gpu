# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.6

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/local/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xzhu/opencl_cmake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xzhu/opencl_cmake

# Include any dependencies generated for this target.
include src/CMakeFiles/cl_gen_header.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/cl_gen_header.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/cl_gen_header.dir/flags.make

src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o: src/CMakeFiles/cl_gen_header.dir/flags.make
src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o: src/cl_gen_header.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/xzhu/opencl_cmake/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o"
	cd /home/xzhu/opencl_cmake/src && /home/kloeckner/mach/x86_64/pool/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cl_gen_header.dir/cl_gen_header.o -c /home/xzhu/opencl_cmake/src/cl_gen_header.cpp

src/CMakeFiles/cl_gen_header.dir/cl_gen_header.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cl_gen_header.dir/cl_gen_header.i"
	cd /home/xzhu/opencl_cmake/src && /home/kloeckner/mach/x86_64/pool/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/xzhu/opencl_cmake/src/cl_gen_header.cpp > CMakeFiles/cl_gen_header.dir/cl_gen_header.i

src/CMakeFiles/cl_gen_header.dir/cl_gen_header.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cl_gen_header.dir/cl_gen_header.s"
	cd /home/xzhu/opencl_cmake/src && /home/kloeckner/mach/x86_64/pool/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/xzhu/opencl_cmake/src/cl_gen_header.cpp -o CMakeFiles/cl_gen_header.dir/cl_gen_header.s

src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o.requires:
.PHONY : src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o.requires

src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o.provides: src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o.requires
	$(MAKE) -f src/CMakeFiles/cl_gen_header.dir/build.make src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o.provides.build
.PHONY : src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o.provides

src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o.provides.build: src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o
.PHONY : src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o.provides.build

# Object files for target cl_gen_header
cl_gen_header_OBJECTS = \
"CMakeFiles/cl_gen_header.dir/cl_gen_header.o"

# External object files for target cl_gen_header
cl_gen_header_EXTERNAL_OBJECTS =

src/cl_gen_header: src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o
src/cl_gen_header: src/CMakeFiles/cl_gen_header.dir/build.make
src/cl_gen_header: src/CMakeFiles/cl_gen_header.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable cl_gen_header"
	cd /home/xzhu/opencl_cmake/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cl_gen_header.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/cl_gen_header.dir/build: src/cl_gen_header
.PHONY : src/CMakeFiles/cl_gen_header.dir/build

src/CMakeFiles/cl_gen_header.dir/requires: src/CMakeFiles/cl_gen_header.dir/cl_gen_header.o.requires
.PHONY : src/CMakeFiles/cl_gen_header.dir/requires

src/CMakeFiles/cl_gen_header.dir/clean:
	cd /home/xzhu/opencl_cmake/src && $(CMAKE_COMMAND) -P CMakeFiles/cl_gen_header.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/cl_gen_header.dir/clean

src/CMakeFiles/cl_gen_header.dir/depend:
	cd /home/xzhu/opencl_cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xzhu/opencl_cmake /home/xzhu/opencl_cmake/src /home/xzhu/opencl_cmake /home/xzhu/opencl_cmake/src /home/xzhu/opencl_cmake/src/CMakeFiles/cl_gen_header.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/cl_gen_header.dir/depend
