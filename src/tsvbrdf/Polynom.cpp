/**
* \file		Polynom.cpp
* \author	Daniel Meister
* \date		2017/11/07
* \brief	Polynom class source file.
*/

#include "Polynom.h"

cv::Mat Polynom::eval(const cv::Mat & t) {
	cv::Mat res(t.size(), t.type(), cv::Scalar(0.0f));
	for (int i = DEGREE; i >= 0; --i)
		res = res.mul(t) + coefs[i];
	return res;
}
