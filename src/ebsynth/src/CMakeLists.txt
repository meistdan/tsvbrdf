# Find CUDA.
find_package(CUDA QUIET REQUIRED)

set(SRC
cudacheck.h
jzq.h
memarray2.h
patchmatch_gpu.h
stb_image.h
stb_image_write.h
texarray2.h
ebsynth.cu
)

# Find includes in corresponding build directories.
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Project name.
set(NAME ebsynth)

# Add all files to library.
cuda_add_library(${NAME} SHARED
  ${SRC}
)

# Add include directories.
set(EBSYNTH_INCLUDE_DIR ../include)
include_directories (SYSTEM
  ${EBSYNTH_INCLUDE_DIR}
)

# Set output directory.
set(BIN_DIR ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(${NAME} PROPERTIES 
	DEBUG_OUTPUT_NAME ${NAME}_d 
	RELEASE_OUTPUT_NAME ${NAME}
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BIN_DIR}
	ARCHIVE_OUTPUT_DIRECTORY ${BIN_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${BIN_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${BIN_DIR}
)

# Define dllexport.
if (WIN32)
	set(EBSYNTH_API __declspec\(dllexport\))
	add_definitions(-DEBSYNTH_API=${EBSYNTH_API})
endif()
	