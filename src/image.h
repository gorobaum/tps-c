#ifndef TPS_IMAGE_H_
#define TPS_IMAGE_H_

#include <vector>
#include <string>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace tps {

class Image {
public:
	Image() {};
	Image(std::string filename) {
		filename_ = filename;
		image = cv::imread(filename_, CV_LOAD_IMAGE_GRAYSCALE);
	  if( !image.data )
      std::cout << " --(!) Error reading images form file" << filename << std::endl;
		dimensions.push_back(image.cols);
		dimensions.push_back(image.rows);
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(95);
	};
	Image(int rows, int cols, const std::string& filename) {
		filename_ = filename;
		image = cv::Mat::zeros(rows, cols, CV_8U);
		dimensions.push_back(image.cols);
		dimensions.push_back(image.rows);
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(95);
	}
	void save() {cv::imwrite(filename_.c_str(), image, compression_params);};
	cv::Mat getImage() {return image;};
	std::vector<int> getDimensions() { return dimensions; };
	template<typename T> void changePixelAt(int col, int row, T value);
	template<typename T> T getPixelAt(int col, int row);
	template<typename T> T bilinearInterpolation(float col, float row);
	template<typename T> T NNInterpolation(float col, float row);
	template<typename T> T* getPixelVector();
	template<typename T> void setPixelVector(T* vector);
private:
	friend class FeatureDetector;
  friend class FeatureGenerator;
  friend class Surf;
	cv::Mat image;
	std::vector<int> dimensions;
	std::string filename_;
	std::vector<int> compression_params;
	int getNearestInteger(float number) {
		if ((number - floor(number)) <= 0.5) return floor(number);
		return floor(number) + 1.0;
	}
};

template<typename T> void Image::changePixelAt(int col, int row, T value) {
	if (col >= 0 && col < dimensions[0]-1 && row >= 0 && row < dimensions[1]-1)
		image.at<T>(col, row) = value;
}

template<typename T>
T Image::getPixelAt(int col, int row) {
	if (col > dimensions[0]-1 || col < 0)
		return 0;
	else if (row > dimensions[1]-1 || row < 0)
		return 0;
	else {
		return image.at<T>(col, row);
	}
}

template<typename T>
T Image::bilinearInterpolation(float col, float row) {
	int u = trunc(col);
	int v = trunc(row);
	uchar pixelOne = getPixelAt<uchar>(u, v);
	uchar pixelTwo = getPixelAt<uchar>(u+1, v);
	uchar pixelThree = getPixelAt<uchar>(u, v+1);
	uchar pixelFour = getPixelAt<uchar>(u+1, v+1);

	T interpolation = (u+1-col)*(v+1-row)*pixelOne
												+ (col-u)*(v+1-row)*pixelTwo 
												+ (u+1-col)*(row-v)*pixelThree
												+ (col-u)*(row-v)*pixelFour;
	return interpolation;
}

template<typename T>
T Image::NNInterpolation(float col, float row) {
	int nearCol = getNearestInteger(col);
	int nearRow = getNearestInteger(row);
	T aux = getPixelAt<uchar>(nearCol, nearRow);
	return aux;
}

template<typename T> 
T* Image::getPixelVector() {
	T* vector = (T*)malloc(dimensions[1]*dimensions[0]*sizeof(T));
  for (int row = 0; row < dimensions[1]; row++) {
    T* ptrRow = image.ptr<T>(row);
    for (int col = 0; col < dimensions[0]; col++) vector[row*dimensions[0]+col] = ptrRow[col];
  }
  return vector;
}

template<typename T> 
void Image::setPixelVector(T* vector) {
for (int row = 0; row < dimensions[1]; row++)
  for (int col = 0; col < dimensions[0]; col++) {
    T newValue = vector[row*dimensions[0]+col];
    changePixelAt(col, row, newValue);
  }
}

} // namespace

#endif