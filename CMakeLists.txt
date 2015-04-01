
cmake_minimum_required (VERSION 2.8)

project (tps)
include_directories (src)
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET REQUIRED)

cuda_add_executable(tps
  src/main.cxx
  src/tps.cxx
  src/surf.cxx
  src/basictps.cxx
  src/paralleltps.cxx
  src/featuregenerator.cxx
	src/cudatps.cu)

target_link_libraries (tps ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode -std=c++11 arch=compute_30,code=sm_30)

list( APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS_DEBUG}")

add_custom_target (
	cleanShit
	COMMAND ${CMAKE_COMMAND} -E remove 
)