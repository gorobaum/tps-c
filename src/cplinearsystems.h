#ifndef TPS_CPLINEARSYSTEMS_H_
#define TPS_CPLINEARSYSTEMS_H_

#include <vector>

#include <opencv2/core/core.hpp>

namespace tps {

class CPLinearSystems {
public:  
  CPLinearSystems(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints) :
    referenceKeypoints_(referenceKeypoints),
    targetKeypoints_(targetKeypoints) {};
  virtual void solveLinearSystems() = 0;
  std::vector<float> getSolutionX() {return solutionX;};
  std::vector<float> getSolutionY() {return solutionY;};
protected:
  virtual void createMatrixA() = 0;
  virtual void createBs() = 0;
  float computeRSquared(float x, float xi, float y, float yi) {return pow(x-xi,2) + pow(y-yi,2);};
  std::vector<cv::Point2f> referenceKeypoints_;
  std::vector<cv::Point2f> targetKeypoints_;
  std::vector<float> solutionX;
  std::vector<float> solutionY;
};

} //namespace

#endif
