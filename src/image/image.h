#ifndef TPS_IMAGE_H_
#define TPS_IMAGE_H_

#include <vector>
#include <string>
#include <cmath>

namespace tps {

class Image {
public:
	Image(std::vector<std::vector<int> > matImage, int width, int height) {
		width_ = width;
		height_ = height;
		image = matImage;
	};
	Image(int width, int height) {
		width_ = width;
		height_ = height;
		image = std::vector<std::vector<int> >(width_, std::vector<int>(height_, 0));
	}
	std::vector<std::vector<int> > getImage() {return image;}
	void save(std::string filename);
	int getWidth() { return width_; }
	int getHeight() { return height_; }
	void changePixelAt(int col, int row, int value);
	int getPixelAt(int col, int row);
	int bilinearInterpolation(float col, float row);
	int NNInterpolation(float col, float row);
	unsigned char* getPixelVector();
	void setPixelVector(unsigned char* vector);
private:
	std::vector<std::vector<int> > image;
	int width_;
	int height_;
	int getNearestInteger(float number) {
		if ((number - std::floor(number)) <= 0.5) return std::floor(number);
		return std::floor(number) + 1.0;
	}
};

} // namespace

#endif