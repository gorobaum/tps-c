#ifndef TPS_CPLINEARSYSTEMS_H_
#define TPS_CPLINEARSYSTEMS_H_

#include <vector>
#include <cmath>

namespace tps {

class CPLinearSystems {
public:  
  CPLinearSystems(std::vector< std::vector<float> > referenceKeypoints, std::vector< std::vector<float> > targetKeypoints) :
    referenceKeypoints_(referenceKeypoints),
    targetKeypoints_(targetKeypoints),
    systemDimension(referenceKeypoints_.size()+4) {};
  std::vector<float> getSolutionX() {return solutionX;};
  std::vector<float> getSolutionY() {return solutionY;};
  std::vector<float> getSolutionZ() {return solutionZ;};
protected:
  virtual void createMatrixA() = 0;
  virtual void createBs() = 0;
  float computeRSquared(float x, float xi, float y, float yi, float z, float zi) {return std::pow(x-xi,2) + std::pow(y-yi,2) + std::pow(z-zi,2);};
  std::vector< std::vector<float> > referenceKeypoints_;
  std::vector< std::vector<float> > targetKeypoints_;
  int systemDimension;
  std::vector<float> solutionX;
  std::vector<float> solutionY;
  std::vector<float> solutionZ;
};

} //namespace

#endif
