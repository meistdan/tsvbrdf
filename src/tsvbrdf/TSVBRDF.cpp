/**
* \file		TSVBRDF.cpp
* \author	Daniel Meister
* \date		2017/11/07
* \brief	TSVBRDF class source file.
*/

#include "TSVBRDF.h"

void TSVBRDF<PolyParameter>::load(const std::string & filepath) {

  // Images.
  cv::Mat img;

  // Diffuse.
  for (int i = 0; i <= Parameter::DEGREE; ++i) {
    img = cv::imread(filepath + "/Diffuse-" + std::to_string(i) + ".exr", cv::IMREAD_UNCHANGED);
    for (int c = 0; c < 3; ++c)
      cv::extractChannel(img, diffuse[c].coefs[i], c);
  }

  // Specular.
  for (int i = 0; i <= Parameter::DEGREE; ++i) {
    img = cv::imread(filepath + "/Specular-" + std::to_string(i) + ".exr", cv::IMREAD_UNCHANGED);
    for (int c = 0; c < 3; ++c)
      cv::extractChannel(img, specular[c].coefs[i], c);
  }

  // Roughness.
  for (int i = 0; i <= Parameter::DEGREE; ++i) {
    img = cv::imread(filepath + "/Roughness-" + std::to_string(i) + ".exr", cv::IMREAD_UNCHANGED);
    cv::extractChannel(img, roughness.coefs[i], 0);
  }

  // Resolution.
  width = img.size().width;
  height = img.size().height;

}

void TSVBRDF<PolyParameter>::save(const std::string & filepath) {
  for (int i = 0; i <= Parameter::DEGREE; ++i) {
    std::vector<cv::Mat> diffuseChannels;
    std::vector<cv::Mat> specularChannels;
    for (int c = 0; c < 3; ++c) {
      diffuseChannels.push_back(diffuse[c].coefs[i]);
      specularChannels.push_back(specular[c].coefs[i]);
    }
    cv::Mat diffs, specs;
    cv::merge(diffuseChannels, diffs);
    cv::merge(specularChannels, specs);
    imwrite(filepath + "/Diffuse-" + std::to_string(i) + ".exr", diffs);
    imwrite(filepath + "/Specular-" + std::to_string(i) + ".exr", specs);
    imwrite(filepath + "/Roughness-" + std::to_string(i) + ".exr", roughness.coefs[i]);
  }
}
