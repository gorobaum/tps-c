#include "image.h"
#include "surf.h"

#include <string>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cout << "Precisa passar o nome dos arquivos coração! \n";    
		return 0;
	}

 	tps::Image referenceImage = tps::Image(argv[1]);
  tps::Image targetImage = tps::Image(argv[2]);

  int minHessian = 400;

  tps::Surf surf = tps::Surf(referenceImage, targetImage, minHessian);
  surf.run(true);

  return 0;
}