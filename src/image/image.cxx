#include "image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void tps::Image::createGridImage(int width, int height, int gridGranuarity) {
  image = std::vector<std::vector<int> >(width_, std::vector<int>(height_, 255));
  float colStep = getWidth()*1.0/(gridGranuarity-1);
  float rowStep = getHeight()*1.0/(gridGranuarity-1);
  for (int col = 0; col <= gridGranuarity; col++)
    for (int i = 0; i < width; i++)
        changePixelAt(getNearestInteger(col*colStep), i, 0);

  for (int row = 0; row <= gridGranuarity; row++)
      for (int i = 0; i < width; i++)
        changePixelAt(i, getNearestInteger(row*rowStep), 0);
}

void tps::Image::changePixelAt(int col, int row, int value) {
  if (col >= 0 && col < width_-1 && row >= 0 && row < height_-1)
    image[col][row] = value;
}

int tps::Image::getPixelAt(int col, int row) {
  if (row > height_-1 || row < 0)
    return 0;
  else if (col > width_-1 || col < 0)
    return 0;
  else {
    return image[col][row];
  }
}

int tps::Image::bilinearInterpolation(float col, float row) {
  int u = trunc(col);
  int v = trunc(row);
  uchar pixelOne = getPixelAt(u, v);
  uchar pixelTwo = getPixelAt(u+1, v);
  uchar pixelThree = getPixelAt(u, v+1);
  uchar pixelFour = getPixelAt(u+1, v+1);

  int interpolation = (u+1-col)*(v+1-row)*pixelOne
                        + (col-u)*(v+1-row)*pixelTwo 
                        + (u+1-col)*(row-v)*pixelThree
                        + (col-u)*(row-v)*pixelFour;
  return interpolation;
}

int tps::Image::NNInterpolation(float col, float row) {
  int nearCol = getNearestInteger(col);
  int nearRow = getNearestInteger(row);
  int aux = getPixelAt(nearCol, nearRow);
  return aux;
}

void tps::Image::save(std::string filename) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  cv::Mat savImage = cv::Mat::zeros(height_, width_, CV_8U);
  for (int col = 0; col < width_; col++)
      for (int row = 0; row < height_; row++)
        savImage.at<uchar>(row, col) = (uchar)image[col][row];

  cv::imwrite(filename.c_str(), savImage, compression_params);
}

uchar* tps::Image::getPixelVector() {
  uchar* vector = (uchar*)malloc(width_*height_*sizeof(uchar));
  for (int col = 0; col < width_; col++)
      for (int row = 0; row < height_; row++) vector[col*height_+row] = (uchar)image[col][row];
  return vector;
}

void tps::Image::setPixelVector(uchar* vector) {
for (int col = 0; col < width_; col++)
  for (int row = 0; row < height_; row++) {
    uchar newValue = vector[col*height_+row];
    changePixelAt(col, row, newValue);
  }
}