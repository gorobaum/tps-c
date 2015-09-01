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
  setSysDim();
  if (twoDimension_) {
    createMatrixA2D();
    createBs2D();
  } else {
    createMatrixA3D();
    createBs3D();
  }

  transferMatrixA();
  transferBs();

  arma::wall_clock timer;
  timer.tic();

  solveLinearSystem(CLSbx, cm.getSolutionX());
  solveLinearSystem(CLSby, cm.getSolutionY());
  solveLinearSystem(CLSbz, cm.getSolutionZ());
  double time = timer.toc();
  std::cout << "Cuda solver execution time: " << time << std::endl;

  solutionX = cm.getHostSolX();
  solutionY = cm.getHostSolY();
  solutionZ = cm.getHostSolZ();

  if (twoDimension_)
    adaptSolutionTo3D();

  for (int i = 0; i < solutionX.size(); i++)
    std::cout << "\t" << solutionX[i] << std::endl;

  freeResources();
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

  cusolverStatus_t cusolver_status;
  cublasStatus_t cublas_status;
  cudaError_t cudaStat;
  cusolverDnHandle_t handle;
  cublasHandle_t cublasH;
  cusolverDnCreate(&handle);
  cublasCreate(&cublasH);

  // step 1: copy A and B to device
  checkCuda(cudaMalloc(&cudaA, systemDimension*systemDimension*sizeof(float)));
  checkCuda(cudaMemcpy(cudaA, CLSA, systemDimension*systemDimension*sizeof(float), cudaMemcpyHostToDevice));
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

  checkCuda(cudaFree(cudaA));
  checkCuda(cudaFree(d_tau));
  checkCuda(cudaFree(devInfo));
  checkCuda(cudaFree(d_work));

  cublasDestroy(cublasH);   
  cusolverDnDestroy(handle);
}

void tps::CudaLinearSystems::transferMatrixA() {
  CLSA = (float*)malloc(systemDimension*systemDimension*sizeof(float));

  for (uint i = 0; i < systemDimension; i++)
    for (uint j = 0; j < systemDimension; j++)
      CLSA[i*systemDimension+j] = matrixA[i][j];

  // for (uint i = 0; i < systemDimension; i++)
  //   for (uint j = 0; j < systemDimension; j++)
  //     std::cout << "CLSA[" << i << "][" << j << "] = " << CLSA[i*systemDimension+j] << std::endl;
}

void tps::CudaLinearSystems::transferBs() {
  CLSbx = (float*)malloc(systemDimension*sizeof(float));
  CLSby = (float*)malloc(systemDimension*sizeof(float));
  CLSbz = (float*)malloc(systemDimension*sizeof(float));
  for (uint i = 0; i < systemDimension; i++) {
    CLSbx[i] = bx[i];
    CLSby[i] = by[i];
    CLSbz[i] = bz[i];
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
  free(CLSA);
  free(CLSbx);
  free(CLSby);
  free(CLSbz);
}