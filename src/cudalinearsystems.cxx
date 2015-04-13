#include "cudalinearsystems.h"

#include <cstdlib>
#include <iostream>
#include <cublas_v2.h>

void tps::CudaLinearSystems::solveLinearSystems() {
  createMatrixA();
  createBs();

  double solverExec = (double)cv::getTickCount();
  solveLinearSystem(bx, floatSolX);
  solveLinearSystem(by, floatSolY);
  solverExec = ((double)cv::getTickCount() - solverExec)/cv::getTickFrequency();
  std::cout << "Cuda solver execution time: " << solverExec << std::endl;

  solutionX = pointerToVector(floatSolX);
  solutionY = pointerToVector(floatSolY);

  freeResources();
  cudaDeviceReset();
}

void tps::CudaLinearSystems::solveLinearSystem(float *B, float *solution) {
  int lwork = 0;
  int info_gpu = 0;

  const int nrhs = 1;
  const float one = 1;

  float *cudaA = NULL;
  float *cudaB = NULL;
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
  cudaMalloc(&cudaA, systemDimension*systemDimension*sizeof(float));
  cudaMalloc(&cudaB, systemDimension*sizeof(float));
  cudaMemcpy(cudaA, A, systemDimension*systemDimension*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaB, B, systemDimension*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_tau, sizeof(float) * systemDimension);
  cudaMalloc((void**)&devInfo, sizeof(int));

  // step 2: query working space of geqrf and ormqr
  cusolver_status = cusolverDnSgeqrf_bufferSize(handle, systemDimension, systemDimension, cudaA, systemDimension, &lwork);

  cudaMalloc((void**)&d_work, sizeof(float) * lwork);

  // step 3: compute QR factorization
  cusolver_status = cusolverDnSgeqrf(handle, systemDimension, systemDimension, cudaA, systemDimension, d_tau, d_work, lwork, devInfo);
  cudaDeviceSynchronize();
  
  // step 4: compute Q^T*B
  cusolver_status = cusolverDnSormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, systemDimension, nrhs, 
    systemDimension, cudaA, systemDimension, d_tau, cudaB, systemDimension, d_work, lwork, devInfo);
  cudaDeviceSynchronize();

  // step 5: compute x = R \ Q^T*B
  cublas_status = cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 
    CUBLAS_DIAG_NON_UNIT, systemDimension, nrhs, &one, cudaA, systemDimension, cudaB, 
    systemDimension);
  cudaDeviceSynchronize();

  cudaMemcpy(solution, cudaB, sizeof(float)*systemDimension, cudaMemcpyDeviceToHost);

  cudaFree(cudaA);
  cudaFree(cudaB);
  cudaFree(d_work);
  cudaFree(d_tau);
  cudaFree(devInfo);

  if (cublasH) cublasDestroy(cublasH);   
  if (handle) cusolverDnDestroy(handle);   
}

void tps::CudaLinearSystems::createMatrixA() {
  A = (float*)malloc(systemDimension*systemDimension*sizeof(float));

 for (uint i = 0; i < referenceKeypoints_.size()+3; i++)
  for (uint j = 0; j < referenceKeypoints_.size()+3; j++)
    A[i*systemDimension+j] = 0.0;

  for (uint j = 0; j < referenceKeypoints_.size(); j++) {
    A[0*systemDimension+j+3] = 1;
    A[1*systemDimension+j+3] = referenceKeypoints_[j].x;
    A[2*systemDimension+j+3] = referenceKeypoints_[j].y;
    A[(j+3)*systemDimension+0] = 1;
    A[(j+3)*systemDimension+1] = referenceKeypoints_[j].x;
    A[(j+3)*systemDimension+2] = referenceKeypoints_[j].y;
  }

  for (uint i = 0; i < referenceKeypoints_.size(); i++)
    for (uint j = 0; j < referenceKeypoints_.size(); j++) {
      float r = computeRSquared(referenceKeypoints_[i].x, referenceKeypoints_[j].x, referenceKeypoints_[i].y, referenceKeypoints_[j].y);
      if (r != 0.0) A[(i+3)*systemDimension+j+3] = r*log(r);
    }
}

void tps::CudaLinearSystems::createBs() {
  bx = (float*)malloc(systemDimension*sizeof(float));
  by = (float*)malloc(systemDimension*sizeof(float));
  floatSolX = (float*)malloc(systemDimension*sizeof(float));
  floatSolY = (float*)malloc(systemDimension*sizeof(float));
  for (uint j = 0; j < 3; j++) {
    bx[j] = 0.0;
    by[j] = 0.0;
  }
  for (uint i = 0; i < referenceKeypoints_.size(); i++) {
    bx[i+3] = targetKeypoints_[i].x;
    by[i+3] = targetKeypoints_[i].y;
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
  free(floatSolX);
  free(floatSolY);
}