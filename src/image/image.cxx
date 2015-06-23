#include "image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void tps::Image::changePixelAt(int x, int y, int z, short value) {
  if (x >= 0 && x < dimensions_[0]-1 && y >= 0 && y < dimensions_[1]-1 && z >= 0 && z <= dimensions_[2]-1)
    image[x][y][z] = value;
}

short tps::Image::getPixelAt(int x, int y, int z) {
  if (x > dimensions_[0]-1 || x < 0)
    return 0;
  else if (y > dimensions_[1]-1 || y < 0)
    return 0;
  else if (z > dimensions_[2]-1 || z < 0)
    return 0;
  else {
    return image[x][y][z];
  }
}

short tps::Image::trilinearInterpolation(float x, float y, float z) {
  int u = trunc(x);
  int v = trunc(y);
  int w = trunc(z);

  int xd = (x - u);
  int yd = (y - v);
  int zd = (z - w);

  uchar c00 = getPixelAt(u, v, w)*(1-xd)+getPixelAt(u+1, v, w)*xd;
  uchar c10 = getPixelAt(u, v+1, w)*(1-xd)+getPixelAt(u+1, v+1, w)*xd;
  uchar c01 = getPixelAt(u, v, w+1)*(1-xd)+getPixelAt(u+1, v, w+1)*xd;
  uchar c11 = getPixelAt(u, v+1, w+1)*(1-xd)+getPixelAt(u+1, v+1, w+1)*xd;

  uchar c0 = c00*(1-yd)+c10*yd;
  uchar c1 = c01*(1-yd)+c11*yd;

  return c0*(1-zd)+c1*zd;
}

short tps::Image::NNInterpolation(float x, float y, float z) {
  int nearX = getNearestInteger(x);
  int nearY = getNearestInteger(y);
  int nearZ = getNearestInteger(z);
  int aux = getPixelAt(nearX, nearY, nearZ);
  return aux;
}

void tps::Image::save(std::string filename) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  cv::Mat savImage = cv::Mat::zeros(dimensions_[1], dimensions_[0], CV_8U);
  for (int x = 0; x < dimensions_[0]; x++)
    for (int y = 0; y < dimensions_[1]; y++)
      savImage.at<uchar>(y, x) = (uchar)image[x][y][0];

  cv::imwrite(filename.c_str(), savImage, compression_params);
}

short* tps::Image::getPixelVector() {
  short* vector = (short*)malloc(dimensions_[0]*dimensions_[1]*dimensions_[2]*sizeof(short));
  for (int z = 0; z < dimensions_[2]; z++)
    for (int x = 0; x < dimensions_[0]; x++)
      for (int y = 0; y < dimensions_[1]; y++)
        vector[z*dimensions_[1]*dimensions_[0]+x*dimensions_[1]+y] = (short)image[x][y][z];
  return vector;
}

void tps::Image::setPixelVector(short* vector) {
  for (int z = 0; z < dimensions_[2]; z++)
    for (int x = 0; x < dimensions_[0]; x++)
      for (int y = 0; y < dimensions_[1]; y++) {
        short newValue = vector[z*dimensions_[1]*dimensions_[0]+x*dimensions_[1]+y];
        changePixelAt(x, y, z, newValue);
      }
}