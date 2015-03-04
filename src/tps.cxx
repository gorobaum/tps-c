#include "image.h"

#include <string>
#include <iostream>
#include <vector>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cout << "Precisa passar o nome dos arquivos coração! \n";    
		return 0;
	}
    Image referenceImage = Image(argv[1]);
    Image targetImage = Image(argv[2]);

    std::cout << referenceImage.getDimensions()[0] << "\n";
    std::cout << referenceImage.getDimensions()[1] << "\n";

    return 0;
}