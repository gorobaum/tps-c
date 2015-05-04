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
	Image(cv::Mat matImage) {
		width_ = matImage.size().width;
		height_ = matImage.size().height;
		image = std::vector<std::vector<int>>(width_, std::vector<int>(height_, 0));
		for (int col = 0; col < width_; col++)
			for (int row = 0; row < height_; row++)
				image[col][row] = (int)matImage.at<uchar>(row, col);
	};
	Image(int width, int height) {
		width_ = width;
		height_ = height;
		image = std::vector<std::vector<int>>(width_, std::vector<int>(height_, 0));
	};
	std::vector<std::vector<int>> getImage() {return image;};
	int getWidth() { return width_; };
	int getHeight() { return height_; };
	void changePixelAt(int col, int row, int value);
	int getPixelAt(int col, int row);
	int bilinearInterpolation(float col, float row);
	int NNInterpolation(float col, float row);
private:
	std::vector<std::vector<int>> image;
	int width_;
	int height_;
	int getNearestInteger(float number) {
		if ((number - floor(number)) <= 0.5) return floor(number);
		return floor(number) + 1.0;
	}
};

} // namespace

#endif