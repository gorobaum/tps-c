#ifndef TPS_IMAGE_H_
#define TPS_IMAGE_H_

#include <vector>
#include <string>
#include <cmath>

namespace tps {

class Image {
public:
	Image(std::vector< std::vector< std::vector<short> > > matImage, std::vector<int> dimensions) :
		dimensions_(dimensions) {
			image = matImage;
	};
	Image(std::vector<int> dimensions) :
		dimensions_(dimensions) {
			image = std::vector< std::vector< std::vector<short> > >(dimensions[0], std::vector< std::vector<short> > (dimensions[1], std::vector<short>(dimensions[2], 0)));
	};
	std::vector< std::vector< std::vector<short> > > getImage() {return image;};
	std::vector<int> getDimensions() { return dimensions_; };
	std::vector<short> getMinMax();
	void changePixelAt(int x, int y, int z, short value);
	short getPixelAt(int x, int y, int z);
	short trilinearInterpolation(float x, float y, float z);
	short NNInterpolation(float x, float y, float z);
	short* getPixelVector();
	float* getFloatPixelVector();
	void setPixelVector(short* vector);
private:
	std::vector< std::vector< std::vector<short> > > image;
	std::vector<int> dimensions_;
	int getNearestInteger(float number) {
		if ((number - std::floor(number)) <= 0.5) return std::floor(number);
		return std::floor(number) + 1.0;
	}
};

} // namespace

#endif