
cmake_minimum_required (VERSION 2.8)

project (tps)
include_directories (src)
find_package(OpenCV REQUIRED)

add_executable(tps
	src/main.cxx
  src/surf.cxx
  src/tps.cxx)

target_link_libraries (tps ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")

add_custom_target (
	cleanShit
	COMMAND ${CMAKE_COMMAND} -E remove 
)


