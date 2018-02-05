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

private:

	std::vector<float> coefs;

public:

	cv::Mat eval(const cv::Mat & t);

	friend class TSVBRDF;

};

#endif /* _POLYNOM_H_ */
