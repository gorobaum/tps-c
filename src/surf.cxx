#include "surf.h"

void tps::Surf::detectFeatures() {
	detector.detect( referenceImage_.image, keypoints_object );
  detector.detect( targetImage_.image, keypoints_scene );
}

void tps::Surf::extractDescriptors() {
  extractor.compute( referenceImage_.image, keypoints_object, descriptors_object );
  extractor.compute( targetImage_.image, keypoints_scene, descriptors_scene );
}

void tps::Surf::matchDescriptors() {
	std::vector<cv::DMatch> matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
  }
}

void tps::Surf::saveFeatureImage() {
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	std::vector<cv::Point2f> obj;
  std::vector<cv::Point2f> scene;

  cv::Mat img_matches;
  drawMatches( referenceImage_.image, keypoints_object, targetImage_.image, keypoints_scene,
               good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  for( uint i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  cv::Mat H = cv::findHomography( obj, scene, CV_RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<cv::Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( referenceImage_.image.cols, 0 );
  obj_corners[2] = cvPoint( referenceImage_.image.cols, referenceImage_.image.rows ); obj_corners[3] = cvPoint( 0, referenceImage_.image.rows );
  std::vector<cv::Point2f> scene_corners(4);

  cv::perspectiveTransform( obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  cv::line( img_matches, scene_corners[0] + cv::Point2f( referenceImage_.image.cols, 0), scene_corners[1] + cv::Point2f( referenceImage_.image.cols, 0), cv::Scalar(0, 255, 0), 4 );
  cv::line( img_matches, scene_corners[1] + cv::Point2f( referenceImage_.image.cols, 0), scene_corners[2] + cv::Point2f( referenceImage_.image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
  cv::line( img_matches, scene_corners[2] + cv::Point2f( referenceImage_.image.cols, 0), scene_corners[3] + cv::Point2f( referenceImage_.image.cols, 0), cv::Scalar( 0, 255, 0), 4 );
  cv::line( img_matches, scene_corners[3] + cv::Point2f( referenceImage_.image.cols, 0), scene_corners[0] + cv::Point2f( referenceImage_.image.cols, 0), cv::Scalar( 0, 255, 0), 4 );

  cv::imwrite("refkeypoints.png", img_matches, compression_params);
}

void tps::Surf::run(bool createFeatureImage) {
	detectFeatures();
	extractDescriptors();
	matchDescriptors();
	if (createFeatureImage) saveFeatureImage();
}