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

	std::vector<float> coefs;

	int degree(void) { return coefs.size() - 1; }
	cv::Mat eval(const cv::Mat & t);

	friend class TSVBRDF;

};

#endif /* _POLYNOM_H_ */
