# Add "opencv"/lib to path
find_package(OpenCV REQUIRED)

# Folder hierarchy.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(HDR
  Globals.h
  Polynom.h
  TSVBRDF.h
)

set(SRC
  main.cpp
  TSVBRDF.cpp
)
# Find includes in corresponding build directories.
set(CMAKE_INCLUDE_CURRENT_DIR ON)

source_group("Header Files\\" FILES ${HDR})
source_group("Source Files\\" FILES ${SRC})

# Project name.
set(NAME tsvbrdf)

# Add all files to excutable.
add_executable(${NAME}
  ${HDR}
  ${SRC}
)

# Add include directories.
include_directories (SYSTEM
  ${EBSYNTH_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries (${NAME}
  ${EBSYNTH_LIBRARY}
  ${OpenCV_LIBS}
)

# Set output directory.
set(BIN_DIR ${CMAKE_SOURCE_DIR}/bin)

# Properties.
set_target_properties(${NAME} PROPERTIES 
	DEBUG_OUTPUT_NAME ${NAME}_d 
	RELEASE_OUTPUT_NAME ${NAME}
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BIN_DIR}
)

if (WIN32)
  set_target_properties(${NAME} PROPERTIES 
    LINK_FLAGS "/NODEFAULTLIB:LIBCMTD /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBC"
  )
endif (WIN32)
