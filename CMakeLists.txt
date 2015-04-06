
cmake_minimum_required (VERSION 2.8)

project (tps)
include_directories (src)
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET REQUIRED)


cuda_add_executable(tps
  src/main.cxx
  src/surf.cxx
  src/featuregenerator.cxx
  src/OPCVlinearsystems.cxx
  src/cudalinearsystems.cxx
  src/basictps.cxx
  src/paralleltps.cxx
	src/cudatps.cu)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

LIST(APPEND CUDA_NVCC_FLAGS )

MESSAGE( "CUDA_NVCC_FLAGS is ${CUDA_CUBLAS_LIBRARIES}" )

# TODO: Fix CUDA_cusolver_LIBRARY in FindCUDA.cmake
target_link_libraries (tps ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} /opt/cuda/lib64/libcusolver.so ${CUDA_CUBLAS_LIBRARIES})

list( APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS_DEBUG}")

add_custom_target (
	cleanShit
	COMMAND ${CMAKE_COMMAND} -E remove 
)