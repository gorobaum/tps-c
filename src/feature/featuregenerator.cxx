#include "featuregenerator.h"

#include <iostream>
#include <vector>
#include <cmath>

#define PI 3.14159265
#define ANG -PI/72.0

void tps::FeatureGenerator::run() {
  std::vector<int> dimensions = referenceImage_.getDimensions();
  gridSizeX = dimensions[0]*percentage_;
  gridSizeY = dimensions[1]*percentage_;
  gridSizeZ = dimensions[2]*percentage_;
  xStep = dimensions[0]*1.0/(gridSizeX-1);
  yStep = dimensions[1]*1.0/(gridSizeY-1);
  zStep = dimensions[2]*1.0/(gridSizeZ-1);
  std::cout << "gridSizeX = " << gridSizeX << std::endl;
  std::cout << "gridSizeY = " << gridSizeY << std::endl;
  std::cout << "gridSizeZ = " << gridSizeZ << std::endl;
  std::cout << "xStep = " << xStep << std::endl;
  std::cout << "yStep = " << yStep << std::endl;
  std::cout << "zStep = " << zStep << std::endl;
  createReferenceImageFeatures();
  createTargetImageFeatures();
}

void tps::FeatureGenerator::createReferenceImageFeatures() {
    for (int z = 0; z < gridSizeZ; z++)
      for (int x = 0; x < gridSizeX; x++)
        for (int y = 0; y < gridSizeY; y++) {
        std::vector<float> newCP;
        newCP.push_back(x*xStep);
        newCP.push_back(y*yStep);
        newCP.push_back(z*zStep);
        referenceKeypoints.push_back(newCP);
      }
}

void tps::FeatureGenerator::createTargetImageFeatures() {
  for (int z = 0; z < gridSizeZ; z++)
    for (int x = 0; x < gridSizeX; x++) 
      for (int y = 0; y < gridSizeY; y++) {
        int pos = z*gridSizeX*gridSizeY+x*gridSizeY+y;
        std::vector<float> referenceCP = referenceKeypoints[pos];
        std::vector<float> newPoint = applyXRotationalDeformationTo(referenceCP[0], referenceCP[1], referenceCP[2], ANG);
        targetKeypoints.push_back(newPoint);
      }
}

std::vector<float> tps::FeatureGenerator::applyXRotationalDeformationTo(float x, float y, float z, float ang) {
  std::vector<int> dimensions = referenceImage_.getDimensions();
  float yC = dimensions[1]/2.0;
  float zC = dimensions[2]/2.0;
  float newY = (y-yC)*std::sin(ang)-(z-zC)*std::cos(ang);
  float newZ = (y-yC)*std::cos(ang)-(z-zC)*std::sin(ang);
  if (x == 90 && y == 108 && z == 90) {
    std::cout << "newX = " << newX << std::endl;
    std::cout << "newY = " << newY << std::endl;
    std::cout << "newZ = " << newZ << std::endl;
  }
  std::vector<float> newPoint;
  newPoint.push_back(x);
  newPoint.push_back(newY);
  newPoint.push_back(newZ);
  return newPoint;
}