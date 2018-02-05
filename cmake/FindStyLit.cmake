if (WIN32)
	find_path(STYLIT_INCLUDE_DIR stylit.h
		lib/stylit
		DOC "The directory where stylit.h resides"
	)
	
	if (CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(STYLIT_LIB_PATHS
			lib/stylit
		)
	elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
		message("WARNING: StyLit 64-bit version is only avaiable!")
	endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
	find_library(STYLIT_LIBRARY
		NAMES stylit
		PATHS
		${STYLIT_LIB_PATHS}
		DOC "The STYLIT library"
	)
	set(STYLIT_LIBRARIES ${STYLIT_LIBRARY})
else (WIN32)
	message("WARNING: StyLit is Windows only!")
endif (WIN32)

if (STYLIT_INCLUDE_DIR)
	set( STYLIT_FOUND 1 CACHE STRING "Set to 1 if STYLIT is found, 0 otherwise")
else (STYLIT_INCLUDE_DIR)
	set( STYLIT_FOUND 0 CACHE STRING "Set to 1 if STYLIT is found, 0 otherwise")
endif (STYLIT_INCLUDE_DIR)

mark_as_advanced( STYLIT_FOUND )
