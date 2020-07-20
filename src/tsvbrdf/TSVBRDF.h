/**
* \file		TSVBRDF.h
* \author	Daniel Meister
* \date		2017/11/07
* \brief	TSVBRDF class header file.
*/

#ifndef _TSVBRDF_H_
#define _TSVBRDF_H_

#include "Polynom.h"

struct Parameter {
  static const int DEGREE = EBSYNTH_PHI_DEGREE;
  virtual cv::Mat eval(float t) = 0;
  virtual void resize(int width, int height, int type) = 0;
  virtual int type(void) = 0;
};

struct PolyParameter : public Parameter {

  cv::Mat coefs[DEGREE + 1];

  cv::Mat eval(float t) {
    cv::Mat res(coefs[0].size(), coefs[0].type(), cv::Scalar(0.0f));
    for (int i = DEGREE; i >= 0; --i)
      res = res.mul(t) + coefs[i];
    return res;
  }

  void resize(int width, int height, int type) {
    for (int i = 0; i <= DEGREE; ++i)
      coefs[i] = cv::Mat(height, width, type);
  }

  int type(void) {
    return coefs[0].type();
  }

};

template <typename Parameter>
class TSVBRDF {

public:

  Parameter diffuse[3];
  Parameter specular[3];
  Parameter roughness;

  int width;
  int height;

  TSVBRDF(void) {}

  TSVBRDF(int width, int height, int type) {
    resize(width, height, type);
  }

  TSVBRDF(const std::string & filepath) {
    load(filepath);
  }

  void resize(int width, int height, int type) {
    this->width = width;
    this->height = height;
    for (int c = 0; c < 3; ++c) {
      diffuse[c].resize(width, height, type);
      specular[c].resize(width, height, type);
    }
    roughness.resize(width, height, type);
  }

  void exportFrames(const std::string & filepath, float frameRate = 0.05f) {
    std::vector<cv::Mat> imgs(3);
    cv::Mat img;
    cv::Mat imgDiff(height, width, diffuse[0].type());
    cv::Mat imgSpec(height, width, specular[0].type());
    cv::Mat imgRoughness(height, width, roughness.type());
    cv::Mat imgSmoothness(height, width, roughness.type());
    cv::Mat imgD(height, width, roughness.type());
    cv::Mat N = (cv::Mat_<float>(3, 1) << 0.0f, 1.0f, 0.0f);
    cv::Mat E = N;
    cv::Mat L = (cv::Mat_<float>(3, 1) << 0.0f, 1.0f, 1.0f);
    L = L / cv::norm(L);
    cv::Mat H = (L + E) * 0.5f;
    H = H / cv::norm(H);
    float dotNL = float(N.dot(L));
    float dotHN = float(N.dot(H));
    float NH2 = acos(dotHN) * acos(dotHN);
    int f = 0;
    for (float t = 0.0f; t <= 1.0f + 1.0e-3f; t += frameRate) {
      imgRoughness = getRoughness(t);
      imgRoughness = cv::max(imgRoughness, 0.0f);
      imgRoughness = cv::min(imgRoughness, 1.0f);
      imgSmoothness = 1.0f - imgRoughness;
      imgD = -NH2 / imgSmoothness.mul(imgSmoothness);
      cv::exp(imgD, imgD);
      for (int c = 0; c < 3; ++c) {
        imgDiff = getDiffuse(t, c);
        imgDiff = cv::max(imgDiff, 0.0f);
        imgSpec = getSpecular(t, c);
        imgSpec = cv::max(imgSpec, 0.0f);
        imgs[c] = (imgDiff + imgSpec.mul(imgD)) * dotNL;
        imgs[c] = imgDiff;
        pow(imgs[c], 0.5f, imgs[c]);
      }
      cv::merge(imgs, img);
	  img = 255.0f * img;
      //cv::resize(img, img, cv::Size(512,512));
      imwrite(filepath + "/" + std::to_string(f++) + ".jpg", img);
    }
  }

  void load(const std::string & filepath);
  void save(const std::string & filepath);

  cv::Mat getDiffuse(float t, int c) {
    return diffuse[c].eval(t);
  }

  cv::Mat getSpecular(float t, int c) {
    return specular[c].eval(t);
  }

  cv::Mat getRoughness(float t) {
    return roughness.eval(t);
  }

  int type(void) {
    return diffuse[0].type();
  }

};

typedef TSVBRDF<PolyParameter> PolyTSVBRDF;

#endif /* _TSVBRDF_H_ */
