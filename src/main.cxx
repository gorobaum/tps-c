#include "image.h"
#include "surf.h"
#include "featuregenerator.h"
#include "basictps.h"
#include "paralleltps.h"
#include "cudatps.h"
#include "cudalinearsystems.h"
#include "OPCVlinearsystems.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>

void readConfigFile(std::string filename, tps::Image& referenceImage, tps::Image& targetImage, std::string& outputName, float& percentage, std::string& extension) {
  std::ifstream infile;
  infile.open(filename.c_str());
  std::string line;
  std::getline(infile, line);
  referenceImage = tps::Image(line);
  std::size_t pos = line.find('.');
  extension = line.substr(pos);
  std::getline(infile, line);
  targetImage = tps::Image(line);
  std::getline(infile, outputName);
  std::getline(infile, line);
  percentage = std::stof(line);
}

int main(int argc, char** argv) {
	if (argc < 1) {
		std::cout << "Precisa passar o arquivo de configuração coração! \n";    
		return 0;
	}
  std::cout << "====================================================\n";
 	tps::Image referenceImage;
  tps::Image targetImage;
  std::string outputName;
  std::string extension;
  float percentage;
  readConfigFile(argv[1], referenceImage, targetImage, outputName, percentage, extension);
  int minHessian = 400;

  double fgExecTime = (double)cv::getTickCount();
  tps::FeatureGenerator fg = tps::FeatureGenerator(referenceImage, targetImage, percentage);
  fg.run(true);
  fgExecTime = ((double)cv::getTickCount() - fgExecTime)/cv::getTickFrequency();
  std::cout << "FeatureGenerator execution time: " << fgExecTime << std::endl;

  int systemDimension = fg.getReferenceKeypoints().size()+3;
  std::vector<int> dimensions = referenceImage.getDimensions();

  std::cout << "Percentage: " << percentage << std::endl;
  std::cout << "Number of control points: " << fg.getReferenceKeypoints().size() << std::endl;
  float totalMemory = fg.getReferenceKeypoints().size()*sizeof(float)*4.0/(1024*1024);
  std::cout << "CPs memory: " << fg.getReferenceKeypoints().size()*sizeof(float)*1.0/(1024*1024) << " MB (4x)" << std::endl;
  totalMemory += dimensions[0]*dimensions[1]*sizeof(double)*2.0/(1024*1024);
  std::cout << "Cuda coord memory: " << dimensions[0]*dimensions[1]*sizeof(double)*1.0/(1024*1024) << " MB (2x)" << std::endl;
  totalMemory += systemDimension*sizeof(float)*2.0/(1024*1024);
  std::cout << "Cuda solution memory: " << systemDimension*sizeof(float)*1.0/(1024*1024) << " MB (2x)" << std::endl;
  totalMemory += fg.getReferenceKeypoints().size()*sizeof(float)*2.0/(1024*1024);
  std::cout << "Cuda CPs memory: " << fg.getReferenceKeypoints().size()*sizeof(float)*1.0/(1024*1024) << " MB (2x)" << std::endl;
  totalMemory += systemDimension*systemDimension*sizeof(float)*1.0/(1024*1024);
  std::cout << "Matrix A memory: " << systemDimension*systemDimension*sizeof(float)*1.0/(1024*1024) << " MB (1x)" << std::endl;
  totalMemory += systemDimension*systemDimension*sizeof(float)*1.0/(1024*1024);
  std::cout << "Vectors B memory: " << systemDimension*sizeof(float)*1.0/(1024*1024) << " MB (4x)" << std::endl;
  std::cout << "Total memory: " << totalMemory << " MB" << std::endl;

  // double opcvExecTime = (double)cv::getTickCount();
  // tps::OPCVLinearSystems opcvlienarSolver = tps::OPCVLinearSystems(fg.getReferenceKeypoints(), fg.getTargetKeypoints());
  // opcvlienarSolver.solveLinearSystems();
  // opcvExecTime = ((double)cv::getTickCount() - opcvExecTime)/cv::getTickFrequency();
  // std::cout << "OpenCV solver execution time: " << opcvExecTime << std::endl;

  // double cudaExecTime = (double)cv::getTickCount();
  // tps::CudaLinearSystems cudalienarSolver = tps::CudaLinearSystems(fg.getReferenceKeypoints(), fg.getTargetKeypoints());
  // cudalienarSolver.solveLinearSystems();
  // cudaExecTime = ((double)cv::getTickCount() - cudaExecTime)/cv::getTickFrequency();
  // std::cout << "Cuda solver execution time: " << cudaExecTime << std::endl;

  // double basicTpsExecTime = (double)cv::getTickCount();
  // tps::BasicTPS tps = tps::BasicTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), targetImage, outputName+"BasicTPS"+extension);
  // tps.run();
  // basicTpsExecTime = ((double)cv::getTickCount() - basicTpsExecTime)/cv::getTickFrequency();
  // std::cout << "Basic TPS execution time: " << basicTpsExecTime << std::endl;

  // double pTpsExecTime = (double)cv::getTickCount();
  // tps::ParallelTPS ptps = tps::ParallelTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), targetImage, outputName+"ParallelTPS"+extension);
  // ptps.run();
  // pTpsExecTime = ((double)cv::getTickCount() - pTpsExecTime)/cv::getTickFrequency();
  // std::cout << "Parallel TPS execution time: " << pTpsExecTime << std::endl;

  if (fg.getReferenceKeypoints().size() < 3700) {
    double OPCVcTpsExecTime = (double)cv::getTickCount();
    tps::CudaTPS OPCVctps = tps::CudaTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), targetImage, outputName+"TPSOPCV"+extension, false);
    OPCVctps.run();
    OPCVcTpsExecTime = ((double)cv::getTickCount() - OPCVcTpsExecTime)/cv::getTickFrequency();
    std::cout << "OPCV Cuda TPS execution time: " << OPCVcTpsExecTime << std::endl;
  }

  double CUDAcTpsExecTime = (double)cv::getTickCount();
  tps::CudaTPS CUDActps = tps::CudaTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), targetImage, outputName+"TPSCUDA"+extension, true);
  CUDActps.run();
  CUDAcTpsExecTime = ((double)cv::getTickCount() - CUDAcTpsExecTime)/cv::getTickFrequency();
  std::cout << "CUDA Cuda TPS execution time: " << CUDAcTpsExecTime << std::endl;
  std::cout << "====================================================\n";
  return 0;
}