#include "image.h"
#include "surf.h"
#include "basictps.h"
#include "paralleltps.h"

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

  std::cout << "Starting Surf execution...\n";
  double surfExecTime = (double)cv::getTickCount();
  tps::Surf surf = tps::Surf(referenceImage, targetImage, minHessian);
  surf.run(true);
  surfExecTime = ((double)cv::getTickCount() - surfExecTime)/cv::getTickFrequency();
  std::cout << "Surf execution time: " << surfExecTime << std::endl;

  std::cout << "Starting Basic TPS execution...\n";
  double basicTpsExecTime = (double)cv::getTickCount();
  tps::BasicTPS tps = tps::BasicTPS(surf.getReferenceKeypoints(), surf.getTargetKeypoints(), targetImage, "regBasic.png");
  tps.run();
  basicTpsExecTime = ((double)cv::getTickCount() - basicTpsExecTime)/cv::getTickFrequency();
  std::cout << "Basic TPS execution time: " << basicTpsExecTime << std::endl;

  std::cout << "Starting Parallel TPS execution...\n";
  double pTpsExecTime = (double)cv::getTickCount();
  tps::ParallelTPS ptps = tps::ParallelTPS(surf.getReferenceKeypoints(), surf.getTargetKeypoints(), targetImage, "regParallel.png");
  ptps.run();
  pTpsExecTime = ((double)cv::getTickCount() - pTpsExecTime)/cv::getTickFrequency();
  std::cout << "Parallel TPS execution time: " << pTpsExecTime << std::endl;

  return 0;
}