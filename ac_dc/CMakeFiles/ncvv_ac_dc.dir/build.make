# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/pybind11_numpy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/pybind11_numpy/build

# Include any dependencies generated for this target.
include CMakeFiles/ncvv_ac_dc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ncvv_ac_dc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ncvv_ac_dc.dir/flags.make

CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.o: CMakeFiles/ncvv_ac_dc.dir/flags.make
CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.o: ../src/numpy_interface.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/pybind11_numpy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.o -c /home/ubuntu/pybind11_numpy/src/numpy_interface.cpp

CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/pybind11_numpy/src/numpy_interface.cpp > CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.i

CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/pybind11_numpy/src/numpy_interface.cpp -o CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.s

# Object files for target ncvv_ac_dc
ncvv_ac_dc_OBJECTS = \
"CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.o"

# External object files for target ncvv_ac_dc
ncvv_ac_dc_EXTERNAL_OBJECTS =

ncvv_ac_dc.cpython-38-x86_64-linux-gnu.so: CMakeFiles/ncvv_ac_dc.dir/src/numpy_interface.cpp.o
ncvv_ac_dc.cpython-38-x86_64-linux-gnu.so: CMakeFiles/ncvv_ac_dc.dir/build.make
ncvv_ac_dc.cpython-38-x86_64-linux-gnu.so: libcode_library.so
ncvv_ac_dc.cpython-38-x86_64-linux-gnu.so: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
ncvv_ac_dc.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpthread.so
ncvv_ac_dc.cpython-38-x86_64-linux-gnu.so: CMakeFiles/ncvv_ac_dc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/pybind11_numpy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ncvv_ac_dc.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ncvv_ac_dc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ncvv_ac_dc.dir/build: ncvv_ac_dc.cpython-38-x86_64-linux-gnu.so

.PHONY : CMakeFiles/ncvv_ac_dc.dir/build

CMakeFiles/ncvv_ac_dc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ncvv_ac_dc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ncvv_ac_dc.dir/clean

CMakeFiles/ncvv_ac_dc.dir/depend:
	cd /home/ubuntu/pybind11_numpy/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/pybind11_numpy /home/ubuntu/pybind11_numpy /home/ubuntu/pybind11_numpy/build /home/ubuntu/pybind11_numpy/build /home/ubuntu/pybind11_numpy/build/CMakeFiles/ncvv_ac_dc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ncvv_ac_dc.dir/depend

Files/ncvv_ac_dc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ncvv_ac_dc.dir/depend

