# Min. CMake version.
cmake_minimum_required(VERSION 2.8.10)

# A name of the project.
project(tsvbrdf)

# Add custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Find external libraries.
find_package(StyLit)

# OpenCV library.
# Add "opencv"/lib to path
find_package(OpenCV REQUIRED)

# Set general compiler flags.
if (WIN32)
	add_definitions(-MP) # Enables multiprocessor compilation.
	add_definitions(-D__WINDOWS__ -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
endif (WIN32)

# Folder hierarchy.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add source files.
add_subdirectory(src)