/**
* \file		TSVBRDF.cpp
* \author	Daniel Meister
* \date		2017/11/07
* \brief	TSVBRDF class source file.
*/

#include "TSVBRDF.h"

TSVBRDF::TSVBRDF() {
}

TSVBRDF::TSVBRDF(int width, int height, int type) {
	resize(width, height, type);
}

TSVBRDF::TSVBRDF(const std::string & filepath) {
	load(filepath);
}

#if 0
void TSVBRDF::load(const std::string & dir) {
	
	// Data path.
	const std::string path = "../data/" + dir + "/SIM/" + dir + "-staf-";

	// Images.
	cv::Mat img;
	const char factorChars[] = { 'A', 'B', 'C', 'D' };
	
	// Border.
	const int BORDER = 20;

	// Kd.
	for (int f = 0; f < 4; ++f) {
		img = cv::imread(path + "Kd-" + factorChars[f] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
		cv::Rect cropRect(BORDER, BORDER, img.size().width - 2 * BORDER, img.size().height - 2 * BORDER);
		img = img(cropRect).clone();
		for (int c = 0; c < 3; ++c)
			cv::extractChannel(img, Kd[c].factors[f], c);
	}

	// Ks.
	for (int f = 0; f < 4; ++f) {
		img = cv::imread(path + "Ks-" + factorChars[f] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
		cv::Rect cropRect(BORDER, BORDER, img.size().width - 2 * BORDER, img.size().height - 2 * BORDER);
		img = img(cropRect).clone();
		cv::extractChannel(img, Ks.factors[f], 0);
	}

	// Sigma.
	for (int f = 0; f < 4; ++f) {
		img = cv::imread(path + "Sigma-" + factorChars[f] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
		cv::Rect cropRect(BORDER, BORDER, img.size().width - 2 * BORDER, img.size().height - 2 * BORDER);
		img = img(cropRect).clone();
		cv::extractChannel(img, sigma.factors[f], 0);
	}

	// Resolution.
	width = img.size().width;
	height = img.size().height;

	// Phi.
	float coef;
	std::string line, value, key;

	// Kd.
	std::ifstream file(path + "Kd-phi.txt");
	if (file.is_open()) {
		int c = 0;
		while (getline(file, line)) {
			std::istringstream ss(line);
			ss >> key;
			if (key.find("pphi") != std::string::npos && !line.empty()) {
				Kd[c].phi.coefs.clear();
				while (true) {
					ss >> coef;
					if (ss.fail())
						break;
					Kd[c].phi.coefs.push_back(coef);
				}
				++c;
			}
		}
		file.close();
	}

	// Ks and sigma.
	file.open(path + "Ks-phi.txt");
	if (file.is_open()) {
		while (getline(file, line)) {
			std::istringstream ss(line);
			ss >> key;
			if (key.find("pphi") != std::string::npos && !line.empty()) {
				TSVBRDF::Parameter & param = key.find("pphi_Ks") != std::string::npos ? Ks : sigma;
				param.phi.coefs.clear();
				while (true) {
					ss >> coef;
					if (ss.fail())
						break;
					param.phi.coefs.push_back(coef);
				}
			}
		}
		file.close();
	}

}
#else
void TSVBRDF::load(const std::string & filepath) {

	// Images.
	cv::Mat img;
	const char factorChars[] = { 'A', 'B', 'C', 'D' };

	// Kd.
	for (int f = 0; f < 4; ++f) {
		img = cv::imread(filepath + "/Kd-" + factorChars[f] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
		for (int c = 0; c < 3; ++c)
			cv::extractChannel(img, Kd[c].factors[f], c);
	}

	// Ks.
	for (int f = 0; f < 4; ++f)
		Ks.factors[f] = cv::imread(filepath + "/Ks-" + factorChars[f] + ".exr", CV_LOAD_IMAGE_UNCHANGED);

	// Sigma.
	for (int f = 0; f < 4; ++f) {
		sigma.factors[f] = cv::imread(filepath + "/Sigma-" + factorChars[f] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
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
		for (int i = 0; i <= Polynom::DEGREE; ++i) {
			ss >> coef;
			Kd[c].phi.coefs[i] = coef;
		}
	}

	// Ks.
	{
		getline(file, line);
		std::istringstream ss(line);
		for (int i = 0; i <= Polynom::DEGREE; ++i) {
			ss >> coef;
			Ks.phi.coefs[i] = coef;
		}
	}

	// Sigma.
	{
		getline(file, line);
		std::istringstream ss(line);
		for (int i = 0; i <= Polynom::DEGREE; ++i) {
			ss >> coef;
			sigma.phi.coefs[i] = coef;
		}
	}

	file.close();

}
#endif

void TSVBRDF::save(const std::string & filepath) {
	const char factorChars[] = { 'A', 'B', 'C', 'D' };
	for (int i = 0; i < 4; ++i) {
		std::vector<cv::Mat> KdChannels;
		for (int c = 0; c < 3; ++c)
			KdChannels.push_back(Kd[c].factors[i]);
		cv::Mat Kds;
		cv::merge(KdChannels, Kds);
		imwrite(filepath + "/Kd-" + factorChars[i] + ".exr", Kds);
		imwrite(filepath + "/Ks-" + factorChars[i] + ".exr", Ks.factors[i]);
		imwrite(filepath + "/Sigma-" + factorChars[i] + ".exr", sigma.factors[i]);
	}
	std::ofstream out(filepath + "/phi.txt");
	for (int c = 0; c < 3; ++c) {
		for (int i = 0; i <= Polynom::DEGREE; ++i)
			out << Kd[c].phi.coefs[i] << " ";
		out << std::endl;
	}
	for (int i = 0; i <= Polynom::DEGREE; ++i)
		out << Ks.phi.coefs[i] << " ";
	out << std::endl;
	for (int i = 0; i <= Polynom::DEGREE; ++i)
		out << sigma.phi.coefs[i] << " ";
	out << std::endl;
	out.close();
}

void TSVBRDF::exportFrames(const std::string & filepath, float frameRate) {
	std::vector<cv::Mat> imgs(3);
	cv::Mat img;
	cv::Mat imgKd(height, width, Kd[0].factors[0].type());
	cv::Mat imgKs(height, width, Ks.factors[0].type());
	cv::Mat imgSigma(height, width, sigma.factors[0].type());
	cv::Mat N = (cv::Mat_<float>(3,1) << 0.0f, 1.0f, 0.0f);
	cv::Mat E = N;
	cv::Mat L = (cv::Mat_<float>(3, 1) << 1.0f, 1.0f, 1.0f);
	cv::Mat H = (L + E) * 0.5f;
	H = H / cv::norm(H);
	float dotNL = N.dot(L);
	float dotEN = N.dot(E);
	float dotHN = N.dot(H);
	float aDotHN2 = acos(dotHN) * acos(dotHN);
	int f = 0;
	for (float t = 0.0f; t <= 1.0f; t += frameRate) {
		imgKs = getKs(t);
		imgKs = cv::max(imgKs, 0.0f);
		imgKs = imgKs / (4.0f * dotNL * dotEN);
		imgSigma = getSigma(t);
		imgSigma = cv::max(imgSigma, 0.0f);
		imgSigma = -imgSigma * aDotHN2;
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

void TSVBRDF::resize(int width, int height, int type) {
	this->width = width;
	this->height = height;
	for (int f = 0; f < 4; ++f) {
		for (int c = 0; c < 3; ++c) {
			Kd[c].factors[f] = cv::Mat(height, width, type);
		}
		Ks.factors[f] = cv::Mat(height, width, type);
		sigma.factors[f] = cv::Mat(height, width, type);
	}
}

cv::Mat TSVBRDF::getKd(float t, int c) {
	return eval(Kd[c], t);
}

cv::Mat TSVBRDF::getKs(float t) {
	return eval(Ks, t);
}

cv::Mat TSVBRDF::getSigma(float t) {
	return eval(sigma, t);
}

cv::Mat TSVBRDF::eval(Parameter & p, float t) {
	cv::Mat tmp = p.phi.eval((t - p.factors[2]).mul(1.0f / p.factors[1]));
	return p.factors[0].mul(tmp) + p.factors[3];
}
