#ifndef _TPS_ITKIMAGELOADER_H_
#define _TPS_ITKIMAGELOADER_H_

#include <itkImage.h>
#include <itkImageIOBase.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>

#include <string>

#include "image.h"

namespace tps {

class ITKImageLoader {
public:
  static tps::Image loadImageData(std::string filename);
private:
  static itk::ImageIOBase::Pointer getImageIO(std::string input);
};

} // namespace

#endif