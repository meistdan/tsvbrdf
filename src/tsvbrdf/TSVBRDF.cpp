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

TSVBRDF::TSVBRDF(const std::string & dir) {
	import(dir);
}

void TSVBRDF::import(const std::string & dir) {
	
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
			cv::extractChannel(img, Kd[c].factors[f], 2 - c);
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

void TSVBRDF::export(const std::string & path, float frameRate) {
	std::vector<cv::Mat> imgs(3);
	cv::Mat img;
	cv::Mat img2(height, width, CV_32FC3);
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
	int f = 0;

	for (float t = 0.0f; t <= 1.0f; t += frameRate) {
		imgKs = getKs(t);
		imgKs = imgKs / (4.0f * dotNL * dotEN);
		imgSigma = getSigma(t);
		imgSigma = cv::max(imgSigma, 0.0f);
		imgSigma = -imgSigma * dotHN * dotHN;
		cv::exp(imgSigma, imgSigma);
		for (int c = 0; c < 3; ++c) {
			imgKd = getKd(t, c);
			imgs[2 - c] = (imgKd + imgKs.mul(imgSigma)) * dotNL;
			//imgs[2 - c] = imgKd;
		}
		cv::merge(imgs, img);
		imwrite(path + "/tvBTF" + std::to_string(f++) + ".jpg", img * 255.0f);
		//imwrite(path + "/tvBTF" + std::to_string(f++) + ".exr", img);
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

cv::Mat TSVBRDF::getKdNormalized(int f, int c) {
	return normalize(Kd[c].factors[f]);
}

cv::Mat TSVBRDF::getKsNormalized(int f) {
	return normalize(Ks.factors[f]);
}

cv::Mat TSVBRDF::getSigmaNormalized(int f) {
	return normalize(sigma.factors[f]);
}

cv::Mat TSVBRDF::eval(Parameter & p, float t) {
	cv::Mat tmp = p.phi.eval((t - p.factors[2]).mul(1.0f / p.factors[1]));
	return p.factors[0].mul(tmp) + p.factors[3];
}

cv::Mat TSVBRDF::normalize(const cv::Mat & mat) {
	double mn, mx;
	cv::minMaxLoc(mat, &mn, &mx);
	return (mat - mn) / (mx - mn);
}

void TSVBRDF::denormalize(const TSVBRDF & source) {
	double mn, mx;
	for (int f = 0; f < 4; ++f) {
		for (int c = 0; c < 3; ++c) {
			cv::minMaxLoc(source.Kd[c].factors[f], &mn, &mx);
			Kd[c].factors[f] = Kd[c].factors[f] * (mx - mn) + mn;
		}
		cv::minMaxLoc(source.Ks.factors[f], &mn, &mx);
		Ks.factors[f] = Ks.factors[f] * (mx - mn) + mn;
		cv::minMaxLoc(source.sigma.factors[f], &mn, &mx);
		sigma.factors[f] = sigma.factors[f] * (mx - mn) + mn;
	}
}
