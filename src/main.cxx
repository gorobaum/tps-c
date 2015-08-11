#include "image/image.h"
#include "feature/featuregenerator.h"
#include "feature/surf.h"
#include "feature/sift.h"
#include "tps/tps.h"
#include "tps/cudatps.h"
#include "tps/basictps.h"
#include "tps/paralleltps.h"
#include "utils/cudamemory.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <itkImage.h>
#include <itkImageIOBase.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>

bool createKeypointImages = true;

void readConfigFile(std::string filename, std::vector< tps::Image >& targetImages,
                    std::vector< cv::Mat >& cvTarImgs, std::vector< std::string >& outputNames, 
                    std::vector< float >& percentages, std::vector< float >& distanceMetrics,
                    std::vector< int >& vnFeatures, std::vector< int >& vnOctavesLayers, 
                    std::vector< double >& vCT, std::vector< double >& vET, std::vector< double >& vSigma) {
  std::ifstream infile;
  infile.open(filename.c_str());
  std::string line;
  
  std::getline(infile, line);
  cv::Mat cvTarImg = cv::imread(line, CV_LOAD_IMAGE_GRAYSCALE);
  std::vector< std::vector<int> > vecImage(cvTarImg.size().width, std::vector<int>(cvTarImg.size().height));
    for (int col = 0; col < cvTarImg.size().width; col++)
      for (int row = 0; row < cvTarImg.size().height; row++)
        vecImage[col][row] = cvTarImg.at<uchar>(row, col);
  tps::Image targetImage = tps::Image(vecImage, cvTarImg.size().width, cvTarImg.size().height);
  cvTarImgs.push_back(cvTarImg);
  targetImages.push_back(targetImage);

  std::string outputName;
  std::getline(infile, outputName);
  outputNames.push_back(outputName);

  std::getline(infile, line);
  float percentage = std::stof(line);
  percentages.push_back(percentage);

  std::getline(infile, line);
  float distanceMetric = std::stof(line);
  distanceMetrics.push_back(distanceMetric);

  std::getline(infile, line);
  int nFeatures = std::stoi(line);
  vnFeatures.push_back(nFeatures);

  std::getline(infile, line);
  int nOctavesLayers = std::stoi(line);
  vnOctavesLayers.push_back(nOctavesLayers);

  std::getline(infile, line);
  double contrastThreshold = std::stod(line);
  vCT.push_back(contrastThreshold);

  std::getline(infile, line);
  double edgeThreshold = std::stod(line);
  vET.push_back(edgeThreshold);

  std::getline(infile, line);
  double sigma = std::stod(line);
  vSigma.push_back(sigma);
}

std::vector< std::vector< float >> addHeight(std::vector< std::vector< float >> newKP, int height) {
  std::vector< std::vector< float >> newKPS;
  for (std::vector<std::vector< float >>::iterator it = newKP.begin() ; it != newKP.end(); ++it) {
    std::vector< float > newPoint = *it;
    newPoint[1] += height;
    newKPS.push_back(newPoint);
  }
  return newKPS;
}

void runFeatureGeneration(tps::Image referenceImage, tps::Image targetImage, float percentage,
    std::string outputName, cv::Mat cvTarImg, cv::Mat cvRefImg, std::vector< std::vector< std::vector<float> > >& referencesKPs, 
    std::vector< std::vector< std::vector<float> > >& targetsKPs, std::string extension, float distanceMetric, 
    int nfeatures, int nOctaveLayers, double contrastThreshold, double edgeThreshold, double sigma) {
    double fgExecTime = (double)cv::getTickCount();

    tps::FeatureGenerator fg = tps::FeatureGenerator(referenceImage, targetImage, percentage);

    // Manual Keypoints
    std::vector< std::vector< float > > refNewKPs = {{27, 71}, {25, 91}, {27, 135}, {58, 178}, {111, 204}, {67, 141}, {58, 87}, {74, 74}, {94, 64}, {120, 64}, {143, 73}, {161, 98}, {172, 124}, {71, 73}, {95, 58}, {123, 56}, {138, 63}};
    std::vector< std::vector< float > > tarNewKPs = {{28, 71}, {33, 110}, {32, 146}, {59, 184}, {113, 210}, {72, 150}, {58, 104}, {75, 82}, {95, 69}, {120, 68}, {142, 81}, {161, 107}, {166, 129}, {71, 73}, {94, 58}, {124, 57}, {148, 71}};

    fg.run();

    fg.addRefKeypoints(refNewKPs);
    fg.addTarKeypoints(tarNewKPs);

    if (createKeypointImages) {
      fg.drawKeypointsImage(cvTarImg, "keypoints-Tar"+outputName+extension);
      fg.drawFeatureImageWithMask(cvRefImg, cvTarImg, "matches-Tar"+outputName+extension, 0);
    }

    fgExecTime = ((double)cv::getTickCount() - fgExecTime)/cv::getTickFrequency();
    referencesKPs.push_back(fg.getReferenceKeypoints());
    targetsKPs.push_back(fg.getTargetKeypoints());
    std::cout << "FeatureGenerator execution time: " << fgExecTime << std::endl;
    std::cout << "============================================" << std::endl;
}

int main(int argc, char** argv) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  if (argc < 1) {
    std::cout << "Precisa passar o arquivo de configuração coração! \n";    
    return 0;
  }

  // Reading the main file
  std::ifstream infile;
  infile.open(argv[1]);

  std::string line; 
  std::getline(infile, line);

  std::size_t pos = line.find('.');
  std::string extension = line.substr(pos);

  cv::Mat cvRefImg = cv::imread(line, CV_LOAD_IMAGE_GRAYSCALE);
  std::vector< std::vector<int> > vecImage(cvRefImg.size().width, std::vector<int>(cvRefImg.size().height));
    for (int col = 0; col < cvRefImg.size().width; col++)
      for (int row = 0; row < cvRefImg.size().height; row++)
        vecImage[col][row] = cvRefImg.at<uchar>(row, col);

  std::getline(infile, line);
  int gridSize = std::stof(line);

  tps::Image referenceImage = tps::Image(vecImage, cvRefImg.size().width, cvRefImg.size().height);
  tps::Image gridImage = tps::Image(referenceImage.getWidth(), referenceImage.getHeight(), gridSize);
  gridImage.save("grid.png");

  std::vector< cv::Mat > cvTarImgs;
  std::vector< tps::Image > targetImages;
  std::vector< std::string > outputNames;
  std::vector< float > percentages;
  std::vector< float > distanceMetrics;
  std::vector< int > vnFeatures;
  std::vector< int > vnOctavesLayers;
  std::vector< double > vCT;
  std::vector< double > vET;
  std::vector< double > vSigma;

  // Reading each iteration configuration file
  int nFiles = 0;
  for (line; std::getline(infile, line); infile.eof(), nFiles++)
    readConfigFile(line, targetImages, cvTarImgs, outputNames, percentages, distanceMetrics, vnFeatures, 
                   vnOctavesLayers, vCT ,vET, vSigma);

  std::vector< std::vector< std::vector<float> > > referencesKPs;
  std::vector< std::vector< std::vector<float> > > targetsKPs;

  // Generating the Control Points for each pair of reference and target images
  std::cout << "============================================" << std::endl;
  for (int i = 0; i < nFiles; i++) {
    std::cout << "Generating the CPs for entry number " << i << std::endl;
    runFeatureGeneration(referenceImage, targetImages[i], percentages[i], outputNames[i], cvTarImgs[i], cvRefImg, 
                         referencesKPs, targetsKPs, extension, distanceMetrics[i], vnFeatures[i], vnOctavesLayers[i], 
                         vCT[i], vET[i], vSigma[i]);
  }

  // Allocating the maximun possible of free memory in the GPU
  std::vector< tps::CudaMemory > cudaMemories;
  // Verifying the total memory occupation inside the GPU

  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = (total - avail)/(1024*1024);
  std::cout << "Device memory used: " << used << "MB" << std::endl;

  std::cout << "============================================" << std::endl;
  for (int i = 0; i < nFiles; i++) {
    int lastI = i;
    while(used <= 1800) {
      std::cout << "Entry number " << i << " will run." << std::endl;
      tps::CudaMemory cm = tps::CudaMemory(targetImages[i].getWidth(), targetImages[i].getHeight(), referencesKPs[i]);
      if (used+cm.memoryEstimation() > 1800) break;
      cm.allocCudaMemory(targetImages[i]);
      cudaMemories.push_back(cm);
      cudaMemGetInfo(&avail, &total);
      used = (total - avail)/(1024*1024);
      std::cout << "Device used memory = " << used << "MB" << std::endl;
      i++;
      if (i >= nFiles) break;
      else std::cout << "--------------------------------------------" << std::endl;
    }

    // Execution of the TPS, both in the Host and in the Device
    std::cout << "============================================" << std::endl;
    for (int j = lastI; j < i; j++) {
      std::cout << "#Execution = " << j << std::endl;
      std::cout << "#Keypoints = " << referencesKPs[j].size() << std::endl;
      std::cout << "#Percentage = " << percentages[j] << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

      double ParallelTpsExecTime = (double)cv::getTickCount();
      tps::ParallelTPS parallelTPS = tps::ParallelTPS(referencesKPs[j], targetsKPs[j], targetImages[j], outputNames[j]+"Parallel"+extension);
      parallelTPS.run();
      ParallelTpsExecTime = ((double)cv::getTickCount() - ParallelTpsExecTime)/cv::getTickFrequency();
      std::cout << "Parallel TPS execution time: " << ParallelTpsExecTime << std::endl;

      std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

      double ParallelGridTpsExecTime = (double)cv::getTickCount();
      tps::ParallelTPS parallelTPSGrid = tps::ParallelTPS(referencesKPs[j], targetsKPs[j], gridImage, outputNames[j]+"ParallelGrid"+extension);
      parallelTPSGrid.run();
      ParallelGridTpsExecTime = ((double)cv::getTickCount() - ParallelGridTpsExecTime)/cv::getTickFrequency();
      std::cout << "ParallelGrid TPS execution time: " << ParallelGridTpsExecTime << std::endl;

      std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

      double CUDAcTpsExecTime = (double)cv::getTickCount();
      tps::CudaTPS CUDActps = tps::CudaTPS(referencesKPs[j], targetsKPs[j], targetImages[j], "Cuda"+outputNames[j]+extension, cudaMemories[j]);
      CUDActps.run();
      CUDAcTpsExecTime = ((double)cv::getTickCount() - CUDAcTpsExecTime)/cv::getTickFrequency();
      std::cout << "Cuda TPS execution time: " << CUDAcTpsExecTime << std::endl;
      cudaMemories[j].freeMemory();

      std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

      tps::CudaMemory cmg = tps::CudaMemory(targetImages[j].getWidth(), targetImages[j].getHeight(), referencesKPs[j]);
      cmg.allocCudaMemory(gridImage);
      double GridcTpsExecTime = (double)cv::getTickCount();
      tps::CudaTPS CUDAtpsGrid = tps::CudaTPS(referencesKPs[j], targetsKPs[j], targetImages[j], "CudaGrid"+outputNames[j]+extension, cmg);
      CUDAtpsGrid.run();
      GridcTpsExecTime = ((double)cv::getTickCount() - GridcTpsExecTime)/cv::getTickFrequency();
      std::cout << "CudaGrid TPS execution time: " << GridcTpsExecTime << std::endl;
      cmg.freeMemory();
      
      std::cout << "============================================" << std::endl;
    }
  }
  return 0;
}