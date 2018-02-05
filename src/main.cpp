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

void spatialPrediction(const std::string & dir) {

	// Source.
	TSVBRDF source(dir);

	// Source guide channels.
	int sn = source.width * source.height;
	int numGuideChannels = 1;
	std::vector<float> sourceGuides(numGuideChannels * sn);

	// Source style channels.
	int numStyleChannels = 20;
	std::vector<cv::Mat> sourceChannels;
#if 1
	for (int f = 0; f < 4; ++f) {
		for (int c = 0; c < 3; ++c)
			sourceChannels.push_back(source.getKdNormalized(f, c));
		sourceChannels.push_back(source.getKsNormalized(f));
		sourceChannels.push_back(source.getSigmaNormalized(f));
	}
#else
	for (int f = 0; f < 4; ++f) {
		for (int c = 0; c < 3; ++c)
			sourceChannels.push_back(source.Kd[c].factors[f]);
		sourceChannels.push_back(source.Ks.factors[f]);
		sourceChannels.push_back(source.sigma.factors[f]);
	}
#endif
	cv::Mat sourceStyles;
	cv::merge(sourceChannels, sourceStyles);

	// Target resolution.
	int targetWidth = 2 * source.width;
	int targetHeight = 2 * source.height;

	// Target guide channels.
	int tn = targetWidth * targetHeight;
	std::vector<float> targetGuides(numGuideChannels * tn);

	// Target style channels => output.
	cv::Mat targetStyles(targetHeight, targetWidth, sourceStyles.type());

	// Style weights.
	const float totalStuleWeight = 1.0f;
	std::vector<float> styleWeights(numStyleChannels);
#if 0
	for (int i = 0; i < numStyleChannels; ++i)
		styleWeights[i] = totalStuleWeight / numStyleChannels;
#else
	for (int f = 0; f < 4; ++f) {
		for (int c = 0; c < 3; ++c)
			styleWeights[c + f * 5] = 1.0f;
		styleWeights[3 + f * 5] = 0.0f;
		styleWeights[4 + f * 5] = 0.0f;
	}
#endif

	// Guide weights.
	std::vector<float> guideWeights(numGuideChannels);
	for (int i = 0; i < numGuideChannels; i++)
		guideWeights[i] = 0.0f;


	// Pyramid levels.
	int patchSize = 5;
	int numPyramidLevels = idealNumPyramidLevels(source.width, source.height, targetWidth, targetHeight, patchSize);
	std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
	std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
	std::vector<int> stopThresholdPerLevel(numPyramidLevels);
	for (int i = 0; i < numPyramidLevels; i++) {
		numSearchVoteItersPerLevel[i] = 8;
		numPatchMatchItersPerLevel[i] = 4;
		stopThresholdPerLevel[i] = 0;
	}

	// StyLit.
	stylitRun(
		STYLIT_BACKEND_CUDA,
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
		5000.0f,
		patchSize,
		STYLIT_VOTEMODE_PLAIN,
		numPyramidLevels,
		numSearchVoteItersPerLevel.data(),
		numPatchMatchItersPerLevel.data(),
		stopThresholdPerLevel.data(),
		0,
		0,
		targetStyles.data
		);

	// Output.
	TSVBRDF target(targetWidth, targetHeight, source.Kd[0].factors[0].type());
	target.Kd[0].phi = source.Kd[0].phi;
	target.Kd[1].phi = source.Kd[1].phi;
	target.Kd[2].phi = source.Kd[2].phi;
	target.Ks.phi = source.Ks.phi;
	target.sigma.phi = source.sigma.phi;

	// Split channels.
	std::vector<cv::Mat> targetChannels;
	for (int f = 0; f < 4; ++f) {
		for (int c = 0; c < 3; ++c)
			targetChannels.push_back(target.Kd[c].factors[f]);
		targetChannels.push_back(target.Ks.factors[f]);
		targetChannels.push_back(target.sigma.factors[f]);
	}
	cv::split(targetStyles, targetChannels);

	// Denormalize.
	target.denormalize(source);

	// Export.
	source.export("../out/reconstruct/" + dir);
	target.export("../out/spatial/" + dir);

}

void temporalPrediction(const std::string & sourceDir, const std::string & targetDir, float t0 = 0.0f) {

	// Source.
	TSVBRDF source(sourceDir);

	// Source guide channels.
	int numGuideChannels = 3;
	std::vector<cv::Mat> sourceGuideChannels;
	sourceGuideChannels.push_back(0.2125f * source.getKd(t0, 0) + 0.7154f * source.getKd(t0, 1) + 0.0721f * source.getKd(t0, 2));
	sourceGuideChannels.push_back(source.getKs(t0));
	sourceGuideChannels.push_back(source.getSigma(t0));
	cv::Mat sourceGuides;
	cv::merge(sourceGuideChannels, sourceGuides);

	// Source style channels.
	int numStyleChannels = 20;
	std::vector<cv::Mat> sourceStyleChannels;
	for (int f = 0; f < 4; ++f) {
		for (int c = 0; c < 3; ++c)
			sourceStyleChannels.push_back(source.Kd[c].factors[f]);
		sourceStyleChannels.push_back(source.Ks.factors[f]);
		sourceStyleChannels.push_back(source.sigma.factors[f]);
	}
	cv::Mat sourceStyles;
	cv::merge(sourceStyleChannels, sourceStyles);

	// Target.
	TSVBRDF target(targetDir);

	// Target guide channels.
	std::vector<cv::Mat> targetGuideChannels;
	targetGuideChannels.push_back(0.2125f * target.getKd(t0, 0) + 0.7154f * target.getKd(t0, 1) + 0.0721f * target.getKd(t0, 2));
	targetGuideChannels.push_back(target.getKs(t0));
	targetGuideChannels.push_back(target.getSigma(t0));
	cv::Mat targetGuides;
	cv::merge(targetGuideChannels, targetGuides);

	// Target style channels => output.
	cv::Mat targetStyles(target.height, target.width, sourceStyles.type());

	// Style weights.
	const float totalStyleWeight = 1.0f;
	std::vector<float> styleWeights(numStyleChannels);
	for (int i = 0; i < numStyleChannels; i++)
		styleWeights[i] = totalStyleWeight / numStyleChannels;

	// Guide weights.
	const float totalGuideWeight = 2.0f;
	std::vector<float> guideWeights(numGuideChannels);
	for (int i = 0; i < numGuideChannels; i++)
		guideWeights[i] = totalGuideWeight / numGuideChannels;

	// Pyramid levels.
	int patchSize = 5;
	int numPyramidLevels = idealNumPyramidLevels(source.width, source.height, target.width, target.height, patchSize);
	std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
	std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
	std::vector<int> stopThresholdPerLevel(numPyramidLevels);
	for (int i = 0; i < numPyramidLevels; i++) {
		numSearchVoteItersPerLevel[i] = 8;
		numPatchMatchItersPerLevel[i] = 4;
		stopThresholdPerLevel[i] = 0;
	}

	// StyLit.
	stylitRun(
		STYLIT_BACKEND_CUDA,
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
		STYLIT_VOTEMODE_PLAIN,
		numPyramidLevels,
		numSearchVoteItersPerLevel.data(),
		numPatchMatchItersPerLevel.data(),
		stopThresholdPerLevel.data(),
		0,
		0,
		targetStyles.data
		);

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
	for (int f = 0; f < 4; ++f) {
		for (int c = 0; c < 3; ++c)
			reconstructChannels.push_back(reconstruct.Kd[c].factors[f]);
		reconstructChannels.push_back(reconstruct.Ks.factors[f]);
		reconstructChannels.push_back(reconstruct.sigma.factors[f]);
	}
	cv::split(targetStyles, reconstructChannels);

	// Errors.
	float KdrMSE = cv::sum((reconstruct.getKd(t0, 0) - target.getKd(t0, 0)).mul(reconstruct.getKd(t0, 0) - target.getKd(t0, 0)))[0];
	float KdgMSE = cv::sum((reconstruct.getKd(t0, 1) - target.getKd(t0, 1)).mul(reconstruct.getKd(t0, 1) - target.getKd(t0, 1)))[0];
	float KdbMSE = cv::sum((reconstruct.getKd(t0, 2) - target.getKd(t0, 2)).mul(reconstruct.getKd(t0, 2) - target.getKd(t0, 2)))[0];
	float KsMSE = cv::sum((reconstruct.getKs(t0) - target.getKs(t0)).mul(reconstruct.getKs(t0) - target.getKs(t0)))[0];
	float sigmaMSE = cv::sum((reconstruct.getSigma(t0) - target.getSigma(t0)).mul(reconstruct.getSigma(t0) - target.getSigma(t0)))[0];

	// Log.
	std::ofstream out("../out/temporal/" + sourceDir + "-" + targetDir + "/out.log");

	// Ranges.
	double mn, mx;
	out << "source:" << std::endl;
	cv::minMaxLoc(source.getKd(t0, 0), &mn, &mx);
	out << "Kdr " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.getKd(t0, 1), &mn, &mx);
	out << "Kdg " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.getKd(t0, 2), &mn, &mx);
	out << "Kdb " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.getKs(t0), &mn, &mx);
	out << "Ks " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.getSigma(t0), &mn, &mx);
	out << "sigma " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(source.Kd[0].factors[0], &mn, &mx);
	out << "Kdr A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Kd[0].factors[1], &mn, &mx);
	out << "Kdr B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Kd[0].factors[2], &mn, &mx);
	out << "Kdr C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Kd[0].factors[3], &mn, &mx);
	out << "Kdr D " << mn << " " << mx << std::endl;
	out << std::endl;
	
	cv::minMaxLoc(source.Kd[1].factors[0], &mn, &mx);
	out << "Kdg A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Kd[1].factors[1], &mn, &mx);
	out << "Kdg B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Kd[1].factors[2], &mn, &mx);
	out << "Kdg C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Kd[1].factors[3], &mn, &mx);
	out << "Kdg D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(source.Kd[2].factors[0], &mn, &mx);
	out << "Kdb A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Kd[2].factors[1], &mn, &mx);
	out << "Kdb B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Kd[2].factors[2], &mn, &mx);
	out << "Kdb C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Kd[2].factors[3], &mn, &mx);
	out << "Kdb D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(source.Ks.factors[0], &mn, &mx);
	out << "Ks A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Ks.factors[1], &mn, &mx);
	out << "Ks B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Ks.factors[2], &mn, &mx);
	out << "Ks C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.Ks.factors[3], &mn, &mx);
	out << "Ks D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(source.sigma.factors[0], &mn, &mx);
	out << "sigma A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.sigma.factors[1], &mn, &mx);
	out << "sigma B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.sigma.factors[2], &mn, &mx);
	out << "sigma C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(source.sigma.factors[3], &mn, &mx);
	out << "sigma D " << mn << " " << mx << std::endl;
	out << std::endl; 
	out << std::endl;

	out << "target:" << std::endl;
	cv::minMaxLoc(target.getKd(t0, 0), &mn, &mx);
	out << "Kdr " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.getKd(t0, 1), &mn, &mx);
	out << "Kdg " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.getKd(t0, 2), &mn, &mx);
	out << "Kdb " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.getKs(t0), &mn, &mx);
	out << "Ks " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.getSigma(t0), &mn, &mx);
	out << "sigma " << mn << " " << mx << std::endl;

	cv::minMaxLoc(target.Kd[0].factors[0], &mn, &mx);
	out << "Kdr A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Kd[0].factors[1], &mn, &mx);
	out << "Kdr B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Kd[0].factors[2], &mn, &mx);
	out << "Kdr C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Kd[0].factors[3], &mn, &mx);
	out << "Kdr D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(target.Kd[1].factors[0], &mn, &mx);
	out << "Kdg A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Kd[1].factors[1], &mn, &mx);
	out << "Kdg B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Kd[1].factors[2], &mn, &mx);
	out << "Kdg C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Kd[1].factors[3], &mn, &mx);
	out << "Kdg D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(target.Kd[2].factors[0], &mn, &mx);
	out << "Kdb A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Kd[2].factors[1], &mn, &mx);
	out << "Kdb B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Kd[2].factors[2], &mn, &mx);
	out << "Kdb C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Kd[2].factors[3], &mn, &mx);
	out << "Kdb D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(target.Ks.factors[0], &mn, &mx);
	out << "Ks A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Ks.factors[1], &mn, &mx);
	out << "Ks B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Ks.factors[2], &mn, &mx);
	out << "Ks C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.Ks.factors[3], &mn, &mx);
	out << "Ks D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(target.sigma.factors[0], &mn, &mx);
	out << "sigma A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.sigma.factors[1], &mn, &mx);
	out << "sigma B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.sigma.factors[2], &mn, &mx);
	out << "sigma C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(target.sigma.factors[3], &mn, &mx);
	out << "sigma D " << mn << " " << mx << std::endl;
	out << std::endl;
	out << std::endl;

	out.close();

//#if 1
//	// Adjust only offset.
//	for (int c = 0; c < 3; ++c)
//		reconstruct.Kd[c].factors[3] = target.getKd(t0, c) - reconstruct.getKd(t0, c) + reconstruct.Kd[c].factors[3];
//	reconstruct.Ks.factors[3] = target.getKs(t0) - reconstruct.getKs(t0) + reconstruct.Ks.factors[3];
//	reconstruct.sigma.factors[3] = target.getSigma(t0) - reconstruct.getSigma(t0) + reconstruct.sigma.factors[3];
//#else
//	// Original paper transfer.
//	for (int c = 0; c < 3; ++c) {
//		reconstruct.Kd[c].factors[0] = (target.getKd(t0, c).mul(1.0f / reconstruct.getKd(t0, c))).mul(reconstruct.Kd[c].factors[0]);
//		reconstruct.Kd[c].factors[3] = (target.getKd(t0, c).mul(1.0f / reconstruct.getKd(t0, c))).mul(reconstruct.Kd[c].factors[3]);
//	}
//	reconstruct.Ks.factors[0] = (target.getKs(t0).mul(1.0f / reconstruct.getKs(t0))).mul(reconstruct.Ks.factors[0]);
//	reconstruct.Ks.factors[3] = (target.getKs(t0).mul(1.0f / reconstruct.getKs(t0))).mul(reconstruct.Ks.factors[3]);
//	reconstruct.sigma.factors[0] = (target.getSigma(t0).mul(1.0f / reconstruct.getSigma(t0))).mul(reconstruct.sigma.factors[0]);
//	reconstruct.sigma.factors[3] = (target.getSigma(t0).mul(1.0f / reconstruct.getSigma(t0))).mul(reconstruct.sigma.factors[3]);
//#endif

	// Export.
	reconstruct.export("../out/temporal/" + sourceDir + "-" + targetDir);

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
	
	if (argc == 2) {
		spatialPrediction(argv[1]);
	}
	
	else if (argc == 3) {
		temporalPrediction(argv[1], argv[2]);
	}

	return 0;
}
