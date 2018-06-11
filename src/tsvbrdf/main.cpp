#include "TSVBRDF.h"

int pyramidLevelSize(int sizeBase, int level) {
	return int(float(sizeBase) * pow(2.0f, -float(level)));
}

int idealNumPyramidLevels(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight, int patchSize) {
	int numLevels = 0;
	for (int level = 32; level >= 0; level--) {
		if (pyramidLevelSize(__min(__min(sourceWidth, sourceHeight),
			__min(targetWidth, targetHeight)), level) >= (2 * patchSize + 1)) {
			numLevels = level + 1;
			break;
		}
	}
	return numLevels;
}

void spatialPrediction(const std::string & srcFilepath, const std::string & outFilepath) {

	// Source.
	TSVBRDF source(srcFilepath);

	// Source guide channels.
	int sn = source.width * source.height;
	int numGuideChannels = 1;
	std::vector<float> sourceGuides(numGuideChannels * sn);

	// Source style channels.
	int numStyleChannels = 25;
	std::vector<cv::Mat> sourceChannels;
	for (int c = 0; c < 3; ++c)
		for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.Kd[c].factors[f]);
	for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.Ks.factors[f]);
	for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.sigma.factors[f]);
	for (int c = 0; c < 3; ++c)
		sourceChannels.push_back(source.getKd(0.0f, c));
	sourceChannels.push_back(source.getKs(0.0f));
	sourceChannels.push_back(source.getSigma(0.0f));
	cv::Mat sourceStyles;
	cv::merge(sourceChannels, sourceStyles);

	// Target resolution.
	int targetWidth = 4 * source.width;
	int targetHeight = 4 * source.height;

	// Target guide channels.
	int tn = targetWidth * targetHeight;
	std::vector<float> targetGuides(numGuideChannels * tn);

	// Target style channels => output.
	cv::Mat targetStyles(targetHeight, targetWidth, sourceStyles.type());

	// Style weights.
	std::vector<float> styleWeights(numStyleChannels);
	for (int i = 0; i < 20; ++i)
		if (i < 12) styleWeights[i] = 1.0f;
		else styleWeights[i] = 0.0f;
	for (int i = 20; i < 25; ++i)
		if (i < 23) styleWeights[i] = 0.0f;
		else styleWeights[i] = 0.0f;

	// Guide weights.
	std::vector<float> guideWeights(numGuideChannels);
	for (int i = 0; i < numGuideChannels; i++)
		guideWeights[i] = 0.0f;

	// Phi.
	std::vector<float> phi;
	for (int c = 0; c < 3; ++c)
		for (int i = 0; i <= Polynom::DEGREE; i++)
			phi.push_back(source.Kd[c].phi.coefs[i]);
	for (int i = 0; i <= Polynom::DEGREE; i++)
		phi.push_back(source.Ks.phi.coefs[i]);
	for (int i = 0; i <= Polynom::DEGREE; i++)
		phi.push_back(source.sigma.phi.coefs[i]);

	// Pyramid levels.
	int patchSize = 5;
	int numPyramidLevels = idealNumPyramidLevels(source.width, source.height, targetWidth, targetHeight, patchSize);
	std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
	std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
	std::vector<int> stopThresholdPerLevel(numPyramidLevels);
	for (int i = 0; i < numPyramidLevels; i++) {
		numSearchVoteItersPerLevel[i] = 8;
		numPatchMatchItersPerLevel[i] = 4;
		stopThresholdPerLevel[i] = 5;
	}

	std::cout << "Start spatial" << std::endl << std::flush;

	// EBSynth.
	ebsynthRun(
		EBSYNTH_BACKEND_CUDA,
		numStyleChannels,
		numGuideChannels,
		source.width,
		source.height,
		sourceStyles.data,
		sourceGuides.data(),
		targetWidth,
		targetHeight,
		targetGuides.data(),
		nullptr,
		styleWeights.data(),
		guideWeights.data(),
		0.075f,
		patchSize,
		EBSYNTH_VOTEMODE_PLAIN,
		numPyramidLevels,
		numSearchVoteItersPerLevel.data(),
		numPatchMatchItersPerLevel.data(),
		stopThresholdPerLevel.data(),
		phi.data(),
		Polynom::DEGREE,
		targetStyles.data
		);

	std::cout << "End spatial" << std::endl << std::flush;

	// Output.
	TSVBRDF target(targetWidth, targetHeight, source.Kd[0].factors[0].type());
	target.Kd[0].phi = source.Kd[0].phi;
	target.Kd[1].phi = source.Kd[1].phi;
	target.Kd[2].phi = source.Kd[2].phi;
	target.Ks.phi = source.Ks.phi;
	target.sigma.phi = source.sigma.phi;

	// Split channels.
	std::vector<cv::Mat> targetChannels;
	for (int c = 0; c < 3; ++c)
		for (int f = 0; f < 4; ++f) targetChannels.push_back(target.Kd[c].factors[f]);
	for (int f = 0; f < 4; ++f) targetChannels.push_back(target.Ks.factors[f]);
	for (int f = 0; f < 4; ++f) targetChannels.push_back(target.sigma.factors[f]);
	for (int i = 0; i < 5; ++i) targetChannels.push_back(cv::Mat(targetHeight, targetWidth, target.Kd[0].factors[0].type()));
	cv::split(targetStyles, targetChannels);

	// Export.
	source.exportFrames(outFilepath + "/reconstruct");
	target.exportFrames(outFilepath + "/spatial");

}

void temporalPrediction(const std::string & srcFilepath, const std::string & tgtFilepath, const std::string & outFilepath, float t0 = 0.0f) {

	// Source.
	TSVBRDF source(srcFilepath);

	// Guide image to be equalized.
	cv::Mat guideImg = 0.2125f * source.getKd(t0, 0) + 0.7154f * source.getKd(t0, 1) + 0.0721f * source.getKd(t0, 2);
	guideImg.convertTo(guideImg, CV_8U, 255.0f);
	equalizeHist(guideImg, guideImg);
	guideImg.convertTo(guideImg, CV_32F, 1.0f / 255.0f);

	// Source guide channels.
	int numGuideChannels = 1;
	cv::Mat sourceGuides = 0.2125f * source.getKd(t0, 0) + 0.7154f * source.getKd(t0, 1) + 0.0721f * source.getKd(t0, 2);
	sourceGuides.convertTo(sourceGuides, CV_8U, 255.0f);
	equalizeHist(sourceGuides, sourceGuides);
	sourceGuides.convertTo(sourceGuides, CV_32F, 1.0f / 255.0f);

	// Source style channels.
	int numStyleChannels = 20;
	std::vector<cv::Mat> sourceChannels;
	for (int c = 0; c < 3; ++c)
		for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.Kd[c].factors[f]);
	for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.Ks.factors[f]);
	for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.sigma.factors[f]);
	cv::Mat sourceStyles;
	cv::merge(sourceChannels, sourceStyles);

	// Target.
	TSVBRDF target(tgtFilepath);

	// Target guide channels.
	cv::Mat targetGuides = 0.2125f * target.getKd(t0, 0) + 0.7154f * target.getKd(t0, 1) + 0.0721f * target.getKd(t0, 2);
	targetGuides.convertTo(targetGuides, CV_8U, 255.0f);
	equalizeHist(targetGuides, targetGuides);
	targetGuides.convertTo(targetGuides, CV_32F, 1.0f / 255.0f);

	// Target style channels => output.
	cv::Mat targetStyles(target.height, target.width, sourceStyles.type());

	// Style weights.
	const float totalStyleWeight = 2.0f;
	std::vector<float> styleWeights(numStyleChannels);
	for (int i = 0; i < numStyleChannels; i++)
		styleWeights[i] = totalStyleWeight / numStyleChannels;

	// Guide weights.
	const float totalGuideWeight = 1.0f;
	std::vector<float> guideWeights(numGuideChannels);
	for (int i = 0; i < numGuideChannels; i++)
		guideWeights[i] = totalGuideWeight / numGuideChannels;

	// Phi.
	std::vector<float> phi;
	for (int c = 0; c < 3; ++c)
		for (int i = 0; i <= Polynom::DEGREE; i++)
			phi.push_back(source.Kd[c].phi.coefs[i]);
	for (int i = 0; i <= Polynom::DEGREE; i++)
		phi.push_back(source.Ks.phi.coefs[i]);
	for (int i = 0; i <= Polynom::DEGREE; i++)
		phi.push_back(source.sigma.phi.coefs[i]);

	// Pyramid levels.
	int patchSize = 5;
	int numPyramidLevels = idealNumPyramidLevels(source.width, source.height, target.width, target.height, patchSize);
	std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
	std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
	std::vector<int> stopThresholdPerLevel(numPyramidLevels);
	for (int i = 0; i < numPyramidLevels; i++) {
		numSearchVoteItersPerLevel[i] = 8;
		numPatchMatchItersPerLevel[i] = 4;
		stopThresholdPerLevel[i] = 5;
	}
	
	std::cout << "Start temporal" << std::endl << std::flush;

	// EBSynth.
	ebsynthRun(
		EBSYNTH_BACKEND_CUDA,
		numStyleChannels,
		numGuideChannels,
		source.width,
		source.height,
		sourceStyles.data,
		sourceGuides.data,
		target.width,
		target.height,
		targetGuides.data,
		nullptr,
		styleWeights.data(),
		guideWeights.data(),
		5000.0f,
		patchSize,
		EBSYNTH_VOTEMODE_PLAIN,
		numPyramidLevels,
		numSearchVoteItersPerLevel.data(),
		numPatchMatchItersPerLevel.data(),
		stopThresholdPerLevel.data(),
		phi.data(),
		Polynom::DEGREE,
		targetStyles.data
		);

	std::cout << "End temporal" << std::endl << std::flush;

	// Output.
	TSVBRDF reconstruct;
	reconstruct.resize(target.width, target.height, source.Kd[0].factors[0].type());

	// Reconstruct.
	reconstruct.Kd[0].phi = source.Kd[0].phi;
	reconstruct.Kd[1].phi = source.Kd[1].phi;
	reconstruct.Kd[2].phi = source.Kd[2].phi;
	reconstruct.Ks.phi = source.Ks.phi;
	reconstruct.sigma.phi = source.sigma.phi;

	// Split channels.
	std::vector<cv::Mat> reconstructChannels;
	for (int c = 0; c < 3; ++c)
		for (int f = 0; f < 4; ++f) reconstructChannels.push_back(reconstruct.Kd[c].factors[f]);
	for (int f = 0; f < 4; ++f) reconstructChannels.push_back(reconstruct.Ks.factors[f]);
	for (int f = 0; f < 4; ++f) reconstructChannels.push_back(reconstruct.sigma.factors[f]);
	cv::split(targetStyles, reconstructChannels);

	// Errors.
	//float KdrMSE = cv::sum((reconstruct.getKd(t0, 0) - target.getKd(t0, 0)).mul(reconstruct.getKd(t0, 0) - target.getKd(t0, 0)))[0];
	//float KdgMSE = cv::sum((reconstruct.getKd(t0, 1) - target.getKd(t0, 1)).mul(reconstruct.getKd(t0, 1) - target.getKd(t0, 1)))[0];
	//float KdbMSE = cv::sum((reconstruct.getKd(t0, 2) - target.getKd(t0, 2)).mul(reconstruct.getKd(t0, 2) - target.getKd(t0, 2)))[0];
	//float KsMSE = cv::sum((reconstruct.getKs(t0) - target.getKs(t0)).mul(reconstruct.getKs(t0) - target.getKs(t0)))[0];
	//float sigmaMSE = cv::sum((reconstruct.getSigma(t0) - target.getSigma(t0)).mul(reconstruct.getSigma(t0) - target.getSigma(t0)))[0];

	// Original paper transfer.
	for (int c = 0; c < 3; ++c) {
		reconstruct.Kd[c].factors[0] = (target.getKd(t0, c).mul(1.0f / reconstruct.getKd(t0, c))).mul(reconstruct.Kd[c].factors[0]);
		reconstruct.Kd[c].factors[3] = (target.getKd(t0, c).mul(1.0f / reconstruct.getKd(t0, c))).mul(reconstruct.Kd[c].factors[3]);
	}
	reconstruct.Ks.factors[0] = (target.getKs(t0).mul(1.0f / reconstruct.getKs(t0))).mul(reconstruct.Ks.factors[0]);
	reconstruct.Ks.factors[3] = (target.getKs(t0).mul(1.0f / reconstruct.getKs(t0))).mul(reconstruct.Ks.factors[3]);
	reconstruct.sigma.factors[0] = (target.getSigma(t0).mul(1.0f / reconstruct.getSigma(t0))).mul(reconstruct.sigma.factors[0]);
	reconstruct.sigma.factors[3] = (target.getSigma(t0).mul(1.0f / reconstruct.getSigma(t0))).mul(reconstruct.sigma.factors[3]);

	// Export.
	reconstruct.exportFrames(outFilepath);

}

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

int main(int argc, char** argv) {
	
	if (argc == 3) {
		spatialPrediction(argv[1], argv[2]);
	}
	
	else if (argc == 4) {
		temporalPrediction(argv[1], argv[2], argv[3]);
	}

	return 0;
}
