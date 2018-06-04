# Min. CMake version.
cmake_minimum_required(VERSION 2.8.10)

# A name of the project.
project(tsvbrdf)

# Set general compiler flags.
if (WIN32)
	add_definitions(-MP) # Enables multiprocessor compilation.
	add_definitions(-D__WINDOWS__ -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
endif (WIN32)

# Ebsynth include dir.
find_path(EBSYNTH_INCLUDE_DIR ebsynth.h
	${CMAKE_SOURCE_DIR}/src/ebsynth/include
	DOC "The directory where ebsynth.h resides"
)

# Ebsynth library.
set(EBSYNTH_LIBRARY ebsynth)

# Add source files.
add_subdirectory(src)
