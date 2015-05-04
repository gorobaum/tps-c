#ifndef TPS_CPLINEARSYSTEMS_H_
#define TPS_CPLINEARSYSTEMS_H_

#include <vector>

#include <opencv2/core/core.hpp>

namespace tps {

class CPLinearSystems {
public:  
  CPLinearSystems(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints) :
    referenceKeypoints_(referenceKeypoints),
    targetKeypoints_(targetKeypoints),
    systemDimension(referenceKeypoints_.size()+3) {};
  virtual void solveLinearSystems() = 0;
  std::vector<float> getSolutionCol() {return solutionCol;};
  std::vector<float> getSolutionRow() {return solutionRow;};
protected:
  virtual void createMatrixA() = 0;
  virtual void createBs() = 0;
  float computeRSquared(float x, float xi, float y, float yi) {return pow(x-xi,2) + pow(y-yi,2);};
  std::vector<cv::Point2f> referenceKeypoints_;
  std::vector<cv::Point2f> targetKeypoints_;
  int systemDimension;
  std::vector<float> solutionCol;
  std::vector<float> solutionRow;
};

} //namespace

#endif
