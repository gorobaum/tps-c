#include "image.h"
#include "surf.h"
#include "tps.h"

#include <string>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>

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

  tps::TPS tps = tps::TPS(surf.getReferenceKeypoints(), surf.getTargetKeypoints(), targetImage);
  tps.run();

  return 0;
}