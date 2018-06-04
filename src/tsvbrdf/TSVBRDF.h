/**
* \file		TSVBRDF.h
* \author	Daniel Meister
* \date		2017/11/07
* \brief	TSVBRDF class header file.
*/

#ifndef _TSVBRDF_H_
#define _TSVBRDF_H_

#include "Polynom.h"

class TSVBRDF {

public:

	struct Parameter {
		Polynom phi;
		cv::Mat factors[4];
	};

	Parameter Kd[3];
	Parameter Ks;
	Parameter sigma;
	
	int width;
	int height;
	
	TSVBRDF(void);
	TSVBRDF(int width, int height, int type);
	TSVBRDF(const std::string & dir);

	void import(const std::string & dir);
	void export(const std::string & path, float frameRate = 0.02f);

	void resize(int width, int height, int type);

	cv::Mat getKd(float t, int c);
	cv::Mat getKs(float t);
	cv::Mat getSigma(float t);

	cv::Mat eval(Parameter & p, float t);
	
};

#endif /* _TSVBRDF_H_ */
