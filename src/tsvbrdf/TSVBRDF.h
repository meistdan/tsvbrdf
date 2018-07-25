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
	TSVBRDF(const std::string & filepath);

	void load(const std::string & filepath);
	void save(const std::string & filepath);

	void exportFrames(const std::string & filepath, float frameRate = 0.02f);

	void resize(int width, int height, int type);

	cv::Mat getKd(float t, int c);
	cv::Mat getKs(float t);
	cv::Mat getSigma(float t);

	cv::Mat getKdStatic(float t, int c);
	cv::Mat getKsStatic(float t);
	cv::Mat getSigmaStatic(float t);

	cv::Mat eval(Parameter & p, float t);
	cv::Mat evalStatic(Parameter & p, float t);
	
	cv::Mat getKdMax(int c);
	cv::Mat getKsMax(void);
	cv::Mat getSigmaMax(void);

	cv::Mat max(Parameter & p, float frameRate = 0.02f);

};

#endif /* _TSVBRDF_H_ */
