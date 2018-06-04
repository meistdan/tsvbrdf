#include "TSVBRDF.h"
#include <windows.h>

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

void dataStatistics(const std::string & dir) {
	
	// Data.
	TSVBRDF material(dir);

	// Output dir.
	const std::string outputDir = "../out/statistics/";
	CreateDirectory(outputDir.c_str(), nullptr);

	// Log.
	std::ofstream out(outputDir + dir + ".log");

	// Extremes.
	double KdrMin = FLT_MAX, KdgMin = FLT_MAX, KdbMin = FLT_MAX, KsMin = FLT_MAX, sigmaMin = FLT_MAX;
	double KdrMax = -FLT_MAX, KdgMax = -FLT_MAX, KdbMax = -FLT_MAX, KsMax = -FLT_MAX, sigmaMax = -FLT_MAX;
	double mn, mx;

	const float DELTA = 0.025f;
	for (float t = 0.0f; t <= 1.0f; t += DELTA) {
		cv::minMaxLoc(material.getKd(t, 0), &mn, &mx); KdrMin = min(KdrMin, mn); KdrMax = max(KdrMax, mx);
		cv::minMaxLoc(material.getKd(t, 1), &mn, &mx); KdgMin = min(KdgMin, mn); KdgMax = max(KdgMax, mx);
		cv::minMaxLoc(material.getKd(t, 2), &mn, &mx); KdbMin = min(KdbMin, mn); KdbMax = max(KdbMax, mx);
		cv::minMaxLoc(material.getKs(t), &mn, &mx); KsMin = min(KsMin, mn); KsMax = max(KsMax, mx);
		cv::minMaxLoc(material.getSigma(t), &mn, &mx); sigmaMin = min(sigmaMin, mn); sigmaMax = max(sigmaMax, mx);
	}
	
	// Print stats.
	out << "Kdr = [" << KdrMin << ", " << KdrMax << "]\n";
	out << "Kdg = [" << KdgMin << ", " << KdgMax << "]\n";
	out << "Kdb = [" << KdbMin << ", " << KdbMax << "]\n";
	out << "Ks = [" << KsMin << ", " << KsMax << "]\n";
	out << "sigma = [" << sigmaMin << ", " << sigmaMax << "]\n";
	out << std::endl;

	out << "Kdr phi degree " << material.Kd[0].phi.degree() << "\n";
	out << "Kdg phi degree " << material.Kd[1].phi.degree() << "\n";
	out << "Kdb phi degree " << material.Kd[2].phi.degree() << "\n";
	out << "Ks phi degree " << material.Ks.phi.degree() << "\n";
	out << "sigma phi degree " << material.sigma.phi.degree() << "\n";
	out << std::endl;

	cv::minMaxLoc(material.Kd[0].factors[0], &mn, &mx);
	out << "Kdr A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Kd[0].factors[1], &mn, &mx);
	out << "Kdr B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Kd[0].factors[2], &mn, &mx);
	out << "Kdr C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Kd[0].factors[3], &mn, &mx);
	out << "Kdr D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(material.Kd[1].factors[0], &mn, &mx);
	out << "Kdg A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Kd[1].factors[1], &mn, &mx);
	out << "Kdg B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Kd[1].factors[2], &mn, &mx);
	out << "Kdg C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Kd[1].factors[3], &mn, &mx);
	out << "Kdg D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(material.Kd[2].factors[0], &mn, &mx);
	out << "Kdb A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Kd[2].factors[1], &mn, &mx);
	out << "Kdb B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Kd[2].factors[2], &mn, &mx);
	out << "Kdb C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Kd[2].factors[3], &mn, &mx);
	out << "Kdb D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(material.Ks.factors[0], &mn, &mx);
	out << "Ks A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Ks.factors[1], &mn, &mx);
	out << "Ks B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Ks.factors[2], &mn, &mx);
	out << "Ks C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.Ks.factors[3], &mn, &mx);
	out << "Ks D " << mn << " " << mx << std::endl;
	out << std::endl;

	cv::minMaxLoc(material.sigma.factors[0], &mn, &mx);
	out << "sigma A " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.sigma.factors[1], &mn, &mx);
	out << "sigma B " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.sigma.factors[2], &mn, &mx);
	out << "sigma C " << mn << " " << mx << std::endl;
	cv::minMaxLoc(material.sigma.factors[3], &mn, &mx);
	out << "sigma D " << mn << " " << mx << std::endl;
	out << std::endl;

	out.close();

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
	for (int c = 0; c < 3; ++c)
		for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.Kd[c].factors[f]);
	for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.Ks.factors[f]);
	for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.sigma.factors[f]);
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
	std::vector<float> styleWeights(numStyleChannels);
	for (int i = 0; i < numStyleChannels; ++i)
		if (i < 12) styleWeights[i] = 1.0f;
		else styleWeights[i] = 0.0f;

	// Guide weights.
	std::vector<float> guideWeights(numGuideChannels);
	for (int i = 0; i < numGuideChannels; i++)
		guideWeights[i] = 0.0f;

	// Phi.
	int phiDegree = EBSYNTH_PHI_DEGREE;
	std::vector<float> phi;
	for (int c = 0; c < 3; ++c) {
		for (int i = 0; i <= phiDegree; i++)
			if (i > source.Kd[c].phi.degree()) phi.push_back(0);
			else phi.push_back(source.Kd[c].phi.coefs[source.Kd[c].phi.degree() - i]);
	}
	for (int i = 0; i <= phiDegree; i++) {
		if (i > source.Ks.phi.degree()) phi.push_back(0);
		else phi.push_back(source.Ks.phi.coefs[source.Ks.phi.degree() - i]);

	}
	for (int i = 0; i <= phiDegree; i++) {
		if (i > source.sigma.phi.degree()) phi.push_back(0);
		else phi.push_back(source.sigma.phi.coefs[source.sigma.phi.degree() - i]);
	}

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
		phiDegree,
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
	cv::split(targetStyles, targetChannels);

	// Output dir.
	const std::string reconstructDir = "../out/reconstruct/";
	const std::string spatialDir = "../out/spatial/";

	// Create dirs.
	CreateDirectory(reconstructDir.c_str(), nullptr);
	CreateDirectory((reconstructDir + dir).c_str(), nullptr);
	CreateDirectory(spatialDir.c_str(), nullptr);
	CreateDirectory((spatialDir + dir).c_str(), nullptr);
	
	// Export.
	source.export(reconstructDir + dir);
	target.export(spatialDir + dir);

}

void temporalPrediction(const std::string & sourceDir, const std::string & targetDir, float t0 = 0.0f) {

	// Source.
	TSVBRDF source(sourceDir);

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
	TSVBRDF target(targetDir);

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
	int phiDegree = EBSYNTH_PHI_DEGREE;
	std::vector<float> phi;
	for (int c = 0; c < 3; ++c) {
		for (int i = 0; i <= phiDegree; i++)
			if (i > source.Kd[c].phi.degree()) phi.push_back(0);
			else phi.push_back(source.Kd[c].phi.coefs[source.Kd[c].phi.degree() - i]);
	}
	for (int i = 0; i <= phiDegree; i++) {
		if (i > source.Ks.phi.degree()) phi.push_back(0);
		else phi.push_back(source.Ks.phi.coefs[source.Ks.phi.degree() - i]);

	}
	for (int i = 0; i <= phiDegree; i++) {
		if (i > source.sigma.phi.degree()) phi.push_back(0);
		else phi.push_back(source.sigma.phi.coefs[source.sigma.phi.degree() - i]);
	}

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
		phiDegree,
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

#if 0
	// Adjust only offset.
	for (int c = 0; c < 3; ++c)
		reconstruct.Kd[c].factors[3] = target.getKd(t0, c) - reconstruct.getKd(t0, c) + reconstruct.Kd[c].factors[3];
	reconstruct.Ks.factors[3] = target.getKs(t0) - reconstruct.getKs(t0) + reconstruct.Ks.factors[3];
	reconstruct.sigma.factors[3] = target.getSigma(t0) - reconstruct.getSigma(t0) + reconstruct.sigma.factors[3];
#else
	// Original paper transfer.
	for (int c = 0; c < 3; ++c) {
		reconstruct.Kd[c].factors[0] = (target.getKd(t0, c).mul(1.0f / reconstruct.getKd(t0, c))).mul(reconstruct.Kd[c].factors[0]);
		reconstruct.Kd[c].factors[3] = (target.getKd(t0, c).mul(1.0f / reconstruct.getKd(t0, c))).mul(reconstruct.Kd[c].factors[3]);
	}
	reconstruct.Ks.factors[0] = (target.getKs(t0).mul(1.0f / reconstruct.getKs(t0))).mul(reconstruct.Ks.factors[0]);
	reconstruct.Ks.factors[3] = (target.getKs(t0).mul(1.0f / reconstruct.getKs(t0))).mul(reconstruct.Ks.factors[3]);
	reconstruct.sigma.factors[0] = (target.getSigma(t0).mul(1.0f / reconstruct.getSigma(t0))).mul(reconstruct.sigma.factors[0]);
	reconstruct.sigma.factors[3] = (target.getSigma(t0).mul(1.0f / reconstruct.getSigma(t0))).mul(reconstruct.sigma.factors[3]);
#endif

	// Output dir.
	const std::string reconstructDir = "../out/temporal/";
	const std::string dir = sourceDir + "-" + targetDir;

	// Create dirs.
	CreateDirectory(reconstructDir.c_str(), nullptr);
	CreateDirectory((reconstructDir + dir).c_str(), nullptr);

	// Export.
	reconstruct.export(reconstructDir + dir);

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
		//dataStatistics(argv[1]);
	}
	
	else if (argc == 3) {
		temporalPrediction(argv[1], argv[2]);
	}

	return 0;
}
