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
    systemDimension(referenceKeypoints_.size()+3) {};
  std::vector<float> getSolutionCol() {return solutionCol;};
  std::vector<float> getSolutionRow() {return solutionRow;};
protected:
  virtual void createMatrixA() = 0;
  virtual void createBs() = 0;
  float computeRSquared(float x, float xi, float y, float yi) {return std::pow(x-xi,2) + std::pow(y-yi,2);};
  std::vector< std::vector<float> > referenceKeypoints_;
  std::vector< std::vector<float> > targetKeypoints_;
  int systemDimension;
  std::vector<float> solutionCol;
  std::vector<float> solutionRow;
};

} //namespace

#endif
