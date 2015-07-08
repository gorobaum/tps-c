#include "cudalinearsystems.h"

#include "cusolver_common.h"
#include "cusolverDn.h"

#include <armadillo>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cublas_v2.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cout << "CUDA Runtime Error: \n" << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}


void tps::CudaLinearSystems::solveLinearSystems(tps::CudaMemory& cm) {
  createMatrixA();
  createBs();

  float deviceMemoryBefore = getDeviceUsedMemory();
  std::cout << "Device memory before the solver = " << deviceMemoryBefore << std::endl;

  arma::wall_clock timer;
  timer.tic();
  solveLinearSystem(bx, cm.getSolutionX());
  solveLinearSystem(by, cm.getSolutionY());
  solveLinearSystem(bz, cm.getSolutionZ());
  double time = timer.toc();
  std::cout << "Cuda solver execution time: " << time << std::endl;

  deviceMemoryBefore = getDeviceUsedMemory();
  std::cout << "Device memory before after the solver = " << deviceMemoryBefore << std::endl;

  solutionX = cm.getHostSolX();
  solutionY = cm.getHostSolY();
  solutionZ = cm.getHostSolZ();

  freeResources();
}

float tps::CudaLinearSystems::getDeviceUsedMemory() {
  size_t avail;
  size_t total;
  cudaMemGetInfo(&avail, &total);
  size_t used = (total - avail)/(1024*1024);
  return used;
}

void tps::CudaLinearSystems::solveLinearSystem(float *B, float *cudaSolution) {
  int lwork = 0;
  int info_gpu = 0;

  const int nrhs = 1;
  const float one = 1;

  float *cudaA = NULL;
  float *d_work = NULL;
  float *d_tau = NULL;
  int *devInfo = NULL;

  float deviceMemoryBefore = getDeviceUsedMemory();
  std::cout << "Device memory before the exec = " << deviceMemoryBefore << std::endl;

  cusolverStatus_t cusolver_status;
  cublasStatus_t cublas_status;
  cudaError_t cudaStat;
  cusolverDnHandle_t handle;
  cublasHandle_t cublasH;
  cusolverDnCreate(&handle);
  cublasCreate(&cublasH);

  // step 1: copy A and B to device
  checkCuda(cudaMalloc(&cudaA, systemDimension*systemDimension*sizeof(float)));

  checkCuda(cudaMemcpy(cudaA, A, systemDimension*systemDimension*sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(cudaSolution, B, systemDimension*sizeof(float), cudaMemcpyHostToDevice));

  checkCuda(cudaMalloc((void**)&d_tau, sizeof(float) * systemDimension));
  checkCuda(cudaMalloc((void**)&devInfo, sizeof(int)));

  // step 2: query working space of geqrf and ormqr
  cusolver_status = cusolverDnSgeqrf_bufferSize(handle, systemDimension, systemDimension, cudaA, systemDimension, &lwork);

  checkCuda(cudaMalloc((void**)&d_work, sizeof(float) * lwork));

  // step 3: compute QR factorization
  cusolver_status = cusolverDnSgeqrf(handle, systemDimension, systemDimension, cudaA, systemDimension, d_tau, d_work, lwork, devInfo);
  cudaDeviceSynchronize();
  
  // step 4: compute Q^T*B
  cusolver_status = cusolverDnSormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, systemDimension, nrhs, 
    systemDimension, cudaA, systemDimension, d_tau, cudaSolution, systemDimension, d_work, lwork, devInfo);
  cudaDeviceSynchronize();

  // step 5: compute x = R \ Q^T*B
  cublas_status = cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 
    CUBLAS_DIAG_NON_UNIT, systemDimension, nrhs, &one, cudaA, systemDimension, cudaSolution, 
    systemDimension);
  cudaDeviceSynchronize();

  deviceMemoryBefore = getDeviceUsedMemory();
  std::cout << "Device memory after the exec= " << deviceMemoryBefore << std::endl;

  checkCuda(cudaFree(cudaA));
  checkCuda(cudaFree(d_tau));
  checkCuda(cudaFree(devInfo));
  checkCuda(cudaFree(d_work));

  deviceMemoryBefore = getDeviceUsedMemory();
  std::cout << "Device memory after the frees = " << deviceMemoryBefore << std::endl;

  cublasDestroy(cublasH);   
  cusolverDnDestroy(handle);

  deviceMemoryBefore = getDeviceUsedMemory();
  std::cout << "Device memory after the destroys = " << deviceMemoryBefore << std::endl;
}

void tps::CudaLinearSystems::createMatrixA() {
  A = (float*)malloc(systemDimension*systemDimension*sizeof(float));

 for (uint i = 0; i < systemDimension; i++)
  for (uint j = 0; j < systemDimension; j++)
    A[i*systemDimension+j] = 0.0;

  for (uint j = 0; j < referenceKeypoints_.size(); j++) {
    A[0*systemDimension+j+4] = 1;
    A[1*systemDimension+j+4] = referenceKeypoints_[j][0];
    A[2*systemDimension+j+4] = referenceKeypoints_[j][1];
    A[3*systemDimension+j+4] = referenceKeypoints_[j][2];
    A[(j+4)*systemDimension+0] = 1;
    A[(j+4)*systemDimension+1] = referenceKeypoints_[j][0];
    A[(j+4)*systemDimension+2] = referenceKeypoints_[j][1];
    A[(j+4)*systemDimension+3] = referenceKeypoints_[j][2];
  }

  for (uint i = 0; i < referenceKeypoints_.size(); i++)
    for (uint j = 0; j < referenceKeypoints_.size(); j++) {
      float r = computeRSquared(referenceKeypoints_[i][0], referenceKeypoints_[j][0], 
                                referenceKeypoints_[i][1], referenceKeypoints_[j][1],
                                referenceKeypoints_[i][2], referenceKeypoints_[j][2]);
      if (r != 0.0) A[(i+4)*systemDimension+j+4] = r*log(r);
    }
}

void tps::CudaLinearSystems::createBs() {
  bx = (float*)malloc(systemDimension*sizeof(float));
  by = (float*)malloc(systemDimension*sizeof(float));
  bz = (float*)malloc(systemDimension*sizeof(float));
  for (uint j = 0; j < 3; j++) {
    bx[j] = 0.0;
    by[j] = 0.0;
    bz[j] = 0.0;
  }
  for (uint i = 0; i < targetKeypoints_.size(); i++) {
    bx[i+4] = targetKeypoints_[i][0];
    by[i+4] = targetKeypoints_[i][1];
    bz[i+4] = targetKeypoints_[i][2];
  }
}

std::vector<float> tps::CudaLinearSystems::pointerToVector(float *pointer) {
  std::vector<float> vector;
  for (int i = 0; i < systemDimension; i++) {
    vector.push_back(pointer[i]);
  }
  return vector;
}

void tps::CudaLinearSystems::freeResources() {
  free(A);
  free(bx);
  free(by);
}