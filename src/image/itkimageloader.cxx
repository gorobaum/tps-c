#include "itkimageloader.h"
#include "image.h"

#include <vector>

typedef itk::ImageIOBase::IOComponentType ScalarPixelType;
typedef itk::Image<short, 3>  ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;

static itk::ImageIOBase::Pointer tps::ITKImageLoader::getImageIO(std::string input) {
  itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(input.c_str(), itk::ImageIOFactory::ReadMode);
 
  imageIO->SetFileName(input);
  imageIO->ReadImageInformation();

  return imageIO;
}

static tps::Image tps::ITKImageLoader::loadImageData(std::string filename) { 
 
  std::vector<int> dimensions;
  itk::ImageIOBase::Pointer imageIO = getImageIO(filename);

  // std::cout << "Pixel Type is " << imageIO->GetComponentTypeAsString(pixelType)<< std::endl;
  // std::cout << "numDimensions: " << numDimensions << std::endl;
  // std::cout << "component size: " << imageIO->GetComponentSize() << std::endl;
  // std::cout << "pixel type (string): " << imageIO->GetPixelTypeAsString(imageIO->GetPixelType()) << std::endl;
  // std::cout << "pixel type: " << imageIO->GetPixelType() << std::endl;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename.c_str());
  reader->Update();
  ImageType::Pointer image = reader->GetOutput();

  ImageType::RegionType region;
  ImageType::IndexType start;
  ImageType::SizeType size;

  for (int i = 0; i < 3; ++i){
    start[i] = 0;
    size[i] = imageIO->GetDimensions(i);
  }

  region.SetSize(size);
  region.SetIndex(start);

  itk::ImageRegionIterator<ImageType> imageIterator(image,region);

  for (int i = 0; i < 3; ++i){
    dimensions.push_back(imageIO->GetDimensions(i));
  }

  std::vector<std::vector<std::vector<short>>> vecImage(dimensions[0], std::vector<std::vector<short>>(dimensions[1], 
                                                        std::vector<short>(dimensions[2], 0)));

  while(!imageIterator.IsAtEnd()){
    ImageType::IndexType index = imageIterator.GetIndex();
    vecImage[index[0]][index[1]][index[2]] = imageIterator.Value();
    ++imageIterator;
  }

  return tps::Image(vecImage, dimensions);
}