
cmake_minimum_required (VERSION 2.8)

project (tps)
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET REQUIRED)
find_package(Armadillo REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
include_directories (src "${ARMADILLO_INCLUDE_DIRS}")

cuda_add_executable(tps
  src/main.cxx
  src/image.cxx
  src/cudamemory.cxx
  src/featuregenerator.cxx
  src/OPCVlinearsystems.cxx
  src/cudalinearsystems.cxx
  src/armalinearsystems.cxx
  src/basictps.cxx
  src/paralleltps.cxx
  src/cudatps.cxx
  src/cutps.cu)

MESSAGE(${ARMADILLO_INCLUDE_DIRS})

LIST(APPEND CUDA_NVCC_FLAGS)

# TODO: Fix CUDA_cusolver_LIBRARY in FindCUDA.cmake

list(APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS_DEBUG}")

target_link_libraries(tps ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} /opt/cuda/lib64/libcusolver.so ${CUDA_CUBLAS_LIBRARIES} ${ARMADILLO_LIBRARIES})

add_custom_target (
	cleanShit
	COMMAND ${CMAKE_COMMAND} -E remove result-* confis/tps.config*
)