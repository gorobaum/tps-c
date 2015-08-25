#include "opcvimagehandler.h"

tps::Image tps::OPCVImageHandler::loadImageData(std::string filename) {
  cv::Mat cvTarImg = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

  std::vector<int> dimensions(3, 0);

  dimensions.push_back(cvTarImg.size().width);
  dimensions.push_back(cvTarImg.size().height);
  dimensions.push_back(1);

  std::vector< std::vector< std::vector<short> > > vecImage(dimensions[0], 
                                                         std::vector< std::vector<short> >(dimensions[1],
                                                         std::vector<short>(dimensions[2], 0)));


    for (int col = 0; col < dimensions[0]; col++)
      for (int row = 0; row < dimensions[1]; row++)
        vecImage[col][row][0] = cvTarImg.at<uchar>(row, col);

  tps::Image(vecImage, dimensions);
}

void tps::OPCVImageHandler::saveImageData(tps::Image resultImage, std::string filename) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  std::vector<int> dimensions = resultImage.getDimensions();

  cv::Mat savImage = cv::Mat::zeros(dimensions[1], dimensions[0], CV_8U);
  for (int col = 0; col < dimensions[0]; col++)
      for (int row = 0; row < dimensions[1]; row++)
        savImage.at<uchar>(row, col) = (uchar)resultImage.getPixelAt(col, row, 0);

  cv::imwrite(filename.c_str(), savImage, compression_params);
}