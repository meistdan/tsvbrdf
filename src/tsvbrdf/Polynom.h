/**
* \file		Polynom.h
* \author	Daniel Meister
* \date		2017/11/07
* \brief	Polynom class header file.
*/

#ifndef _POLYNOM_H_
#define _POLUNOM_H_

#include "Globals.h"

template <int DEGREE>
class Polynom {
public:

	float coefs[DEGREE + 1];
  cv::Mat eval(const cv::Mat & t) {
    cv::Mat res(t.size(), t.type(), cv::Scalar(0.0f));
    for (int i = DEGREE; i >= 0; --i)
      res = res.mul(t) + coefs[i];
    return res;
  }

  float eval(float t) {
    float res = 0.0f;
    for (int i = DEGREE; i >= 0; --i)
      res = res * t + coefs[i];
    return res;
  }

};

#endif /* _POLYNOM_H_ */
