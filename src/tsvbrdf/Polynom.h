/**
* \file		Polynom.h
* \author	Daniel Meister
* \date		2017/11/07
* \brief	Polynom class header file.
*/

#ifndef _POLYNOM_H_
#define _POLUNOM_H_

#include "Globals.h"

class Polynom {
public:

	static const int DEGREE = 6;
	float coefs[DEGREE + 1];

	cv::Mat eval(const cv::Mat & t);
	float eval(float t);

};

#endif /* _POLYNOM_H_ */
