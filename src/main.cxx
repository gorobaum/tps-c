#include "image.h"
#include "surf.h"
#include "basictps.h"
#include "paralleltps.h"
#include "cudatps.h"

#include <string>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cout << "Precisa passar o nome dos arquivos coração! \n";    
		return 0;
	}

 	tps::Image referenceImage = tps::Image(argv[1]);
  tps::Image targetImage = tps::Image(argv[2]);
  std::string outputName = argv[3];

  int minHessian = 400;

  tps::Surf surf = tps::Surf(referenceImage, targetImage, minHessian);
  surf.run(true);

  double basicTpsExecTime = (double)cv::getTickCount();
  tps::BasicTPS tps = tps::BasicTPS(surf.getReferenceKeypoints(), surf.getTargetKeypoints(), targetImage, "regBasic.png");
  tps.run();
  basicTpsExecTime = ((double)cv::getTickCount() - basicTpsExecTime)/cv::getTickFrequency();
  std::cout << "Basic TPS execution time: " << basicTpsExecTime << std::endl;

  double pTpsExecTime = (double)cv::getTickCount();
  tps::ParallelTPS ptps = tps::ParallelTPS(surf.getReferenceKeypoints(), surf.getTargetKeypoints(), targetImage, "regParallel.png");
  ptps.run();
  pTpsExecTime = ((double)cv::getTickCount() - pTpsExecTime)/cv::getTickFrequency();
  std::cout << "Parallel TPS execution time: " << pTpsExecTime << std::endl;

  double cTpsExecTime = (double)cv::getTickCount();
  tps::CudaTPS ctps = tps::CudaTPS(surf.getReferenceKeypoints(), surf.getTargetKeypoints(), targetImage, "regCuda.png");
  ctps.run();
  cTpsExecTime = ((double)cv::getTickCount() - cTpsExecTime)/cv::getTickFrequency();
  std::cout << "Cuda TPS execution time: " << cTpsExecTime << std::endl;

  return 0;
}