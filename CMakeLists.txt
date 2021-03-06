cmake_minimum_required (VERSION 2.8)

project (tps)

find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)
find_package(CUDA REQUIRED)
find_package(ITK REQUIRED)

include(${ITK_USE_FILE})

include_directories(src "${ARMADILLO_INCLUDE_DIRS}")

cuda_add_executable(tps
  src/main.cxx

  src/image/image.cxx
  src/image/itkimagehandler.cxx

  src/utils/cudamemory.cxx

  src/feature/featuregenerator.cxx

  # src/linearsystem/OPCVlinearsystems.cxx
  src/linearsystem/cudalinearsystems.cxx
  src/linearsystem/armalinearsystems.cxx

  src/tps/basictps.cxx
  src/tps/paralleltps.cxx
  src/tps/cudatps.cxx
  src/tps/cutps.cu
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fpermissive -include cstddef")

# TODO: Fix CUDA_cusolver_LIBRARY in FindCUDA.cmake
target_link_libraries(tps 
    ${ITK_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT} 
    ${OpenCV_LIBS} 
    /opt/cuda/lib64/libcusolver.so 
    ${CUDA_CUBLAS_LIBRARIES} 
    ${ARMADILLO_LIBRARIES})

add_custom_target (
	cleanResults
	COMMAND ${CMAKE_COMMAND} -E remove result-*
)

add_custom_target(
  cleanConfigurations
  COMMAND ${CMAKE_COMMAND} -E remove configuration/tps.config* tps.configs
)