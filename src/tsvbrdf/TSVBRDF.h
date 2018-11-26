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
  static const int DEGREE = 6;
  virtual cv::Mat eval(float t) = 0;
  virtual cv::Mat evalStatic(float t) = 0;
  virtual void resize(int width, int height, int type) = 0;
  virtual int type(void) = 0;
};

struct STAFParameter : public Parameter {

  Polynom<DEGREE> phi;
  cv::Mat factors[4];

  cv::Mat eval(float t) {
    cv::Mat tmp = phi.eval((t - factors[2]).mul(1.0f / factors[1]));
    return factors[0].mul(tmp) + factors[3];
  }

  cv::Mat evalStatic(float t) {
    cv::Mat tmp = phi.eval(cv::Mat(factors[0].size(), factors[0].type(), cv::Scalar(t)));
    return factors[0].mul(tmp) + factors[3];
  }

  void resize(int width, int height, int type) {
    for (int f = 0; f < 4; ++f)
      factors[f] = cv::Mat(height, width, type);
  }

  int type(void) {
    return factors[0].type();
  }

};

struct PolyParameter : public Parameter {

  cv::Mat coefs[DEGREE + 1];

  cv::Mat eval(float t) {
    cv::Mat res(coefs[0].size(), coefs[0].type(), cv::Scalar(t));
    for (int i = DEGREE; i >= 0; --i)
      res = res.mul(t) + coefs[i];
    return res;
  }

  cv::Mat evalStatic(float t) {
    return eval(t);
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

  Parameter Kd[3];
  Parameter Ks;
  Parameter sigma;

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
    for (int c = 0; c < 3; ++c)
      Kd[c].resize(width, height, type);
    Ks.resize(width, height, type);
    sigma.resize(width, height, type);
  }

  void exportFrames(const std::string & filepath, float frameRate = 0.02f) {
    std::vector<cv::Mat> imgs(3);
    cv::Mat img;
    cv::Mat imgKd(height, width, Kd[0].type());
    cv::Mat imgKs(height, width, Ks.type());
    cv::Mat imgSigma(height, width, sigma.type());
    cv::Mat N = (cv::Mat_<float>(3, 1) << 0.0f, 1.0f, 0.0f);
    cv::Mat E = N;
    cv::Mat L = (cv::Mat_<float>(3, 1) << 0.0f, 1.0f, 0.0f);
    //cv::Mat L = (cv::Mat_<float>(3, 1) << 0.0f, 1.0f, 1.0f);
    L = L / cv::norm(L);
    cv::Mat H = (L + E) * 0.5f;
    H = H / cv::norm(H);
    float dotNL = float(N.dot(L));
    float dotEN = float(N.dot(E));
    float dotHN = float(N.dot(H));
    float aDotHN = acos(dotHN);
    int f = 0;
    for (float t = 0.0f; t <= 1.0f; t += frameRate) {
      imgKs = getKs(t);
      imgKs = cv::max(imgKs, 0.0f);
      imgKs = imgKs / (4.0f * dotNL * dotEN);
      imgSigma = getSigma(t);
      imgSigma = cv::max(imgSigma, 0.0f);
      imgSigma = -imgSigma * aDotHN * aDotHN;
      cv::exp(imgSigma, imgSigma);
      for (int c = 0; c < 3; ++c) {
        imgKd = getKd(t, c);
        imgKd = cv::max(imgKd, 0.0f);
        imgs[c] = (imgKd + imgKs.mul(imgSigma)) * dotNL;
      }
      cv::merge(imgs, img);
      imwrite(filepath + "/" + std::to_string(f++) + ".jpg", img * 255.0f);
    }
  }

  void load(const std::string & filepath);
  void save(const std::string & filepath);

  cv::Mat getKd(float t, int c) {
    return Kd[c].eval(t);
  }

  cv::Mat getKs(float t) {
    return Ks.eval(t);
  }

  cv::Mat getSigma(float t) {
    return sigma.eval(t);
  }

  cv::Mat getKdStatic(float t, int c) {
    return Kd[c].evalStatic(t);
  }

  cv::Mat getKsStatic(float t) {
    return Ks.evalStatic(t);
  }

  cv::Mat getSigmaStatic(float t) {
    return sigma.evalStatic(t);
  }

};

typedef TSVBRDF<STAFParameter> STAFTSVBRDF;
typedef TSVBRDF<PolyParameter> PolyTSVBRDF;

#endif /* _TSVBRDF_H_ */
