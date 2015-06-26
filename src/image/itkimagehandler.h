#ifndef _TPS_ITKIMAGEHANDLER_H_
#define _TPS_ITKIMAGEHANDLER_H_

#include <itkImage.h>
#include <itkImageIOBase.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>

#include <string>

#include "image.h"

namespace tps {

class ITKImageHandler {
public:
  static tps::Image loadImageData(std::string filename);
  static void saveImageData(tps::Image resultImage, std::string filename);
private:
  static itk::ImageIOBase::Pointer getImageIO(std::string input);
};

} // namespace

#endif