/**
* \file		TSVBRDF.cpp
* \author	Daniel Meister
* \date		2017/11/07
* \brief	TSVBRDF class source file.
*/

#include "TSVBRDF.h"

void TSVBRDF<STAFParameter>::load(const std::string & filepath) {

  // Images.
  cv::Mat img;
  const char factorChars[] = { 'A', 'B', 'C', 'D' };

  // Kd.
  for (int i = 0; i < 4; ++i) {
    img = cv::imread(filepath + "/Kd-" + factorChars[i] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
    for (int c = 0; c < 3; ++c)
      cv::extractChannel(img, Kd[c].factors[i], c);
  }

  // Ks.
  for (int i = 0; i < 4; ++i) {
    img = cv::imread(filepath + "/Ks-" + factorChars[i] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
    cv::extractChannel(img, Ks.factors[i], 0);
  }

  // Sigma.
  for (int i = 0; i < 4; ++i) {
    img = cv::imread(filepath + "/Sigma-" + factorChars[i] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
    cv::extractChannel(img, sigma.factors[i], 0);
  }

  // Resolution.
  width = img.size().width;
  height = img.size().height;

  // Phi.
  float coef;
  std::string line, value;
  std::ifstream file(filepath + "/phi.txt");

  // Kd.
  for (int c = 0; c < 3; ++c) {
    getline(file, line);
    std::istringstream ss(line);
    for (int i = 0; i <= Parameter::DEGREE; ++i) {
      ss >> coef;
      Kd[c].phi.coefs[i] = coef;
    }
  }

  // Ks.
  {
    getline(file, line);
    std::istringstream ss(line);
    for (int i = 0; i <= Parameter::DEGREE; ++i) {
      ss >> coef;
      Ks.phi.coefs[i] = coef;
    }
  }

  // Sigma.
  {
    getline(file, line);
    std::istringstream ss(line);
    for (int i = 0; i <= Parameter::DEGREE; ++i) {
      ss >> coef;
      sigma.phi.coefs[i] = coef;
    }
  }

  file.close();

}

void TSVBRDF<STAFParameter>::save(const std::string & filepath) {
  const char factorChars[] = { 'A', 'B', 'C', 'D' };
  for (int i = 0; i < 4; ++i) {
    std::vector<cv::Mat> KdChannels;
    std::vector<cv::Mat> KsChannels;
    std::vector<cv::Mat> sigmaChannels;
    for (int c = 0; c < 3; ++c) {
      KdChannels.push_back(Kd[c].factors[i]);
      KsChannels.push_back(Ks.factors[i]);
      sigmaChannels.push_back(sigma.factors[i]);
    }
    cv::Mat Kds;
    cv::Mat Kss;
    cv::Mat sigmas;
    cv::merge(KdChannels, Kds);
    cv::merge(KsChannels, Kss);
    cv::merge(sigmaChannels, sigmas);
    imwrite(filepath + "/Kd-" + factorChars[i] + ".exr", Kds);
    imwrite(filepath + "/Ks-" + factorChars[i] + ".exr", Kss);
    imwrite(filepath + "/Sigma-" + factorChars[i] + ".exr", sigmas);
  }
  std::ofstream out(filepath + "/phi.txt");
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i <= Parameter::DEGREE; ++i)
      out << Kd[c].phi.coefs[i] << " ";
    out << std::endl;
  }
  for (int i = 0; i <= Parameter::DEGREE; ++i)
    out << Ks.phi.coefs[i] << " ";
  out << std::endl;
  for (int i = 0; i <= Parameter::DEGREE; ++i)
    out << sigma.phi.coefs[i] << " ";
  out << std::endl;
  out.close();
}

void TSVBRDF<PolyParameter>::load(const std::string & filepath) {
  
  // Images.
  cv::Mat img;

  // Kd.
  for (int i = 0; i <= Parameter::DEGREE; ++i) {
    img = cv::imread(filepath + "/Kd-" + std::to_string(i) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
    for (int c = 0; c < 3; ++c)
      cv::extractChannel(img, Kd[c].coefs[i], c);
  }

  // Ks.
  for (int i = 0; i <= Parameter::DEGREE; ++i) {
    img = cv::imread(filepath + "/Ks-" + std::to_string(i) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
    cv::extractChannel(img, Ks.coefs[i], 0);
  }

  // Sigma.
  for (int i = 0; i <= Parameter::DEGREE; ++i) {
    img = cv::imread(filepath + "/Sigma-" + std::to_string(i) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
    cv::extractChannel(img, sigma.coefs[i], 0);
  }

  // Resolution.
  width = img.size().width;
  height = img.size().height;
  
}

void TSVBRDF<PolyParameter>::save(const std::string & filepath) {
  for (int i = 0; i < Parameter::DEGREE; ++i) {
    std::vector<cv::Mat> KdChannels;
    std::vector<cv::Mat> KsChannels;
    std::vector<cv::Mat> sigmaChannels;
    for (int c = 0; c < 3; ++c) {
      KdChannels.push_back(Kd[c].coefs[i]);
      KsChannels.push_back(Ks.coefs[i]);
      sigmaChannels.push_back(sigma.coefs[i]);
    }
    cv::Mat Kds;
    cv::Mat Kss;
    cv::Mat sigmas;
    cv::merge(KdChannels, Kds);
    cv::merge(KsChannels, Kss);
    cv::merge(sigmaChannels, sigmas);
    imwrite(filepath + "/Kd-" + std::to_string(i) + ".exr", Kds);
    imwrite(filepath + "/Ks-" + std::to_string(i) + ".exr", Kss);
    imwrite(filepath + "/Sigma-" + std::to_string(i) + ".exr", sigmas);
  }
}
