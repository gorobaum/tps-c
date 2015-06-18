cmake_minimum_required (VERSION 2.8)

project (tps)

find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)
find_package(CUDA REQUIRED)
find_package(ITK REQUIRED)

include(${ITK_USE_FILE}) # see vtkModuleAPI.cmake:110


OPTION(BUILD_SHARED_LIBS "Build with shared libraries."
       ${VTK_BUILD_SHARED_LIBS})

include_directories(src "${ARMADILLO_INCLUDE_DIRS}")

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
  src/cutps.cu
  )

# TODO: Fix CUDA_cusolver_LIBRARY in FindCUDA.cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fpermissive -include cstddef")

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