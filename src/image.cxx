#include "image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void tps::Image::changePixelAt(int col, int row, int slice, int value) {
  if (col >= 0 && col < width_-1 && row >= 0 && row < height_-1 && slice >= 0 && slice <= slices_-1)
    image[col][row][slice] = value;
}

int tps::Image::getPixelAt(int col, int row, int slice) {
  if (row > height_-1 || row < 0)
    return 0;
  else if (col > width_-1 || col < 0)
    return 0;
  else if (slice > slices_-1 || slice < 0)
    return 0;
  else {
    return image[col][row][slice];
  }
}

int tps::Image::trilinearInterpolation(float col, float row, float slice) {
  int u = trunc(col);
  int v = trunc(row);
  int w = trunc(slice);

  int xd = (col - u);
  int yd = (row - v);
  int zd = (slice - w);

  uchar c00 = getPixelAt(u, v, w)*(1-xd)+getPixelAt(u+1, v, w)*xd;
  uchar c10 = getPixelAt(u, v+1, w)*(1-xd)+getPixelAt(u+1, v+1, w)*xd;
  uchar c01 = getPixelAt(u, v, w+1)*(1-xd)+getPixelAt(u+1, v, w+1)*xd;
  uchar c11 = getPixelAt(u, v+1, w+1)*(1-xd)+getPixelAt(u+1, v+1, w+1)*xd;

  uchar c0 = c00*(1-yd)+c10*yd;
  uchar c1 = c01*(1-yd)+c11*yd;

  return c0*(1-zd)+c1*zd;
}

int tps::Image::NNInterpolation(float col, float row, float slice) {
  int nearCol = getNearestInteger(col);
  int nearRow = getNearestInteger(row);
  int nearSlice = getNearestInteger(slice);
  int aux = getPixelAt(nearCol, nearRow, nearSlice);
  return aux;
}

void tps::Image::save(std::string filename) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  cv::Mat savImage = cv::Mat::zeros(height_, width_, CV_8U);
  for (int col = 0; col < width_; col++)
    for (int row = 0; row < height_; row++)
      savImage.at<uchar>(row, col) = (uchar)image[col][row][0];

  cv::imwrite(filename.c_str(), savImage, compression_params);
}

uchar* tps::Image::getPixelVector() {
  uchar* vector = (uchar*)malloc(width_*height_*sizeof(uchar));
  for (int slice = 0; slice < slices_; slice++)
    for (int col = 0; col < width_; col++)
      for (int row = 0; row < height_; row++)
        vector[slice*height_*width_+col*height_+row] = (uchar)image[col][row][slice];
  return vector;
}

void tps::Image::setPixelVector(uchar* vector) {
  for (int slice = 0; slice < slices_; slice++)
    for (int col = 0; col < width_; col++)
      for (int row = 0; row < height_; row++) {
        uchar newValue = vector[slice*height_*width_+col*height_+row];
        changePixelAt(col, row, slice, newValue);
      }
}
