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
  PolyTSVBRDF source(srcFilepath);

	// Source guide channels.
	int sn = source.width * source.height;
	int numGuideChannels = 1;
	std::vector<float> sourceGuides(numGuideChannels * sn);

	// Source style channels.
	int numStyleChannels = 5 * (Parameter::DEGREE + 1);
	std::vector<cv::Mat> sourceChannels;
	for (int c = 0; c < 3; ++c)
		for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.Kd[c].coefs[i]);
	for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.Ks.coefs[i]);
	for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.sigma.coefs[i]);
	cv::Mat sourceStyles;
	cv::merge(sourceChannels, sourceStyles);

	// Target resolution.
  const int SIZE_MULT = 4;
	int targetWidth = SIZE_MULT * source.width;
	int targetHeight = SIZE_MULT * source.height;

	// Target guide channels.
	int tn = targetWidth * targetHeight;
	std::vector<float> targetGuides(numGuideChannels * tn);

	// Target style channels => output.
	cv::Mat targetStyles(targetHeight, targetWidth, sourceStyles.type());

	// Style weights.
	std::vector<float> styleWeights(numStyleChannels);
  for (int i = 0; i < 5 * (Parameter::DEGREE + 1); ++i) {
    if (i < 4 * (Parameter::DEGREE + 1)) styleWeights[i] = 1.0f;
    else styleWeights[i] = 0.0f;
  }
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
    Parameter::DEGREE,
		targetStyles.data
		);

	// Output.
  PolyTSVBRDF target(targetWidth, targetHeight, source.type());

	// Split channels.
	std::vector<cv::Mat> targetChannels;
	for (int c = 0; c < 3; ++c)
		for (int i = 0; i <= Parameter::DEGREE; ++i) targetChannels.push_back(target.Kd[c].coefs[i]);
	for (int i = 0; i <= Parameter::DEGREE; ++i) targetChannels.push_back(target.Ks.coefs[i]);
	for (int i = 0; i <= Parameter::DEGREE; ++i) targetChannels.push_back(target.sigma.coefs[i]);
	cv::split(targetStyles, targetChannels);
  
	// Export.
	source.exportFrames(srcFilepath + "/images");
	target.exportFrames(outFilepath + "/images");
	target.save(outFilepath);

}

void spatialPredictionRef(const std::string & srcFilepath, const std::string & outFilepath) {

	//// Source.
 // STAFTSVBRDF source(srcFilepath);

	//// Source guide channels.
	//int sn = source.width * source.height;
	//int numGuideChannels = 1;
	//std::vector<float> sourceGuides(numGuideChannels * sn);

	//// Source style channels.
	//int numStyleChannels = 25;
	//std::vector<cv::Mat> sourceChannels;
	//for (int c = 0; c < 3; ++c)
	//	for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.Kd[c].factors[f]);
	//for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.Ks.factors[f]);
	//for (int f = 0; f < 4; ++f) sourceChannels.push_back(source.sigma.factors[f]);
	//for (int c = 0; c < 3; ++c)
	//	sourceChannels.push_back(source.getKdStatic(0.0f, c));
	//sourceChannels.push_back(source.getKsStatic(0.0f));
	//cv::Mat sigma = source.getSigmaStatic(0.0f);
	//cv::sqrt(sigma, sigma);
	//sigma = 1.0f / sigma;
	//sourceChannels.push_back(sigma);
	//cv::Mat sourceStyles;
	//cv::merge(sourceChannels, sourceStyles);

	//// Target resolution.
	//int targetWidth = 4 * source.width;
	//int targetHeight = 4 * source.height;

	//// Target guide channels.
	//int tn = targetWidth * targetHeight;
	//std::vector<float> targetGuides(numGuideChannels * tn);

	//// Target style channels => output.
	//cv::Mat targetStyles(targetHeight, targetWidth, sourceStyles.type());

	//// Style weights.
	//std::vector<float> styleWeights(numStyleChannels);
	//for (int i = 0; i < 20; ++i)
	//	styleWeights[i] = 0.0f;
	//for (int i = 20; i < 25; ++i)
	//	styleWeights[i] = 1.0f;

	//// Guide weights.
	//std::vector<float> guideWeights(numGuideChannels);
	//for (int i = 0; i < numGuideChannels; i++)
	//	guideWeights[i] = 0.0f;

	//// Phi.
	//std::vector<float> phi;
	//for (int c = 0; c < 3; ++c)
	//	for (int i = 0; i <= Parameter::DEGREE; i++)
	//		phi.push_back(source.Kd[c].phi.coefs[i]);
	//for (int i = 0; i <= Parameter::DEGREE; i++)
	//	phi.push_back(source.Ks.phi.coefs[i]);
	//for (int i = 0; i <= Parameter::DEGREE; i++)
	//	phi.push_back(source.sigma.phi.coefs[i]);

	//// Pyramid levels.
	//int patchSize = 5;
	//int numPyramidLevels = idealNumPyramidLevels(source.width, source.height, targetWidth, targetHeight, patchSize);
	//std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
	//std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
	//std::vector<int> stopThresholdPerLevel(numPyramidLevels);
	//for (int i = 0; i < numPyramidLevels; i++) {
	//	numSearchVoteItersPerLevel[i] = 8;
	//	numPatchMatchItersPerLevel[i] = 4;
	//	stopThresholdPerLevel[i] = 0;
	//}

	//// EBSynth.
	//ebsynthRun(
	//	EBSYNTH_BACKEND_CUDA,
	//	numStyleChannels,
	//	numGuideChannels,
	//	source.width,
	//	source.height,
	//	sourceStyles.data,
	//	sourceGuides.data(),
	//	targetWidth,
	//	targetHeight,
	//	targetGuides.data(),
	//	nullptr,
	//	styleWeights.data(),
	//	guideWeights.data(),
	//	0.075f,
	//	patchSize,
	//	EBSYNTH_VOTEMODE_PLAIN,
	//	numPyramidLevels,
	//	numSearchVoteItersPerLevel.data(),
	//	numPatchMatchItersPerLevel.data(),
	//	stopThresholdPerLevel.data(),
	//	phi.data(),
 //   Parameter::DEGREE,
	//	targetStyles.data
	//);

	//// Output.
 // STAFTSVBRDF target(targetWidth, targetHeight, source.Kd[0].factors[0].type());
	//target.Kd[0].phi = source.Kd[0].phi;
	//target.Kd[1].phi = source.Kd[1].phi;
	//target.Kd[2].phi = source.Kd[2].phi;
	//target.Ks.phi = source.Ks.phi;
	//target.sigma.phi = source.sigma.phi;

	//// Split channels.
	//std::vector<cv::Mat> targetChannels;
	//for (int c = 0; c < 3; ++c)
	//	for (int f = 0; f < 4; ++f) targetChannels.push_back(target.Kd[c].factors[f]);
	//for (int f = 0; f < 4; ++f) targetChannels.push_back(target.Ks.factors[f]);
	//for (int f = 0; f < 4; ++f) targetChannels.push_back(target.sigma.factors[f]);
	//for (int i = 0; i < 5; ++i) targetChannels.push_back(cv::Mat(targetHeight, targetWidth, target.Kd[0].factors[0].type()));
	//cv::split(targetStyles, targetChannels);

	//// Image0 and Image1.
	//cv::Mat Kd0[3], Kd1[3];
	//cv::Mat Ks0, Ks1;
	//cv::Mat sigma0, sigma1;
	//float KdPhi0[3], KdPhi1[3];
	//float KsPhi0, KsPhi1;
	//float sigmaPhi0, sigmaPhi1;
	//for (int c = 0; c < 3; ++c) {
	//	Kd0[c] = target.getKd(0.0f, c);
	//	Kd1[c] = target.getKd(1.0f, c);
	//	KdPhi0[c] = target.Kd[c].phi.eval(0.0f);
	//	KdPhi1[c] = target.Kd[c].phi.eval(1.0f);
	//}
	//Ks0 = target.getKs(0.0f);
	//Ks1 = target.getKs(1.0f);
	//KsPhi0 = target.Ks.phi.eval(0.0f);
	//KsPhi1 = target.Ks.phi.eval(1.0f);
	//sigma0 = target.getSigma(0.0f);
	//sigma1 = target.getSigma(1.0f);
	//sigmaPhi0 = target.sigma.phi.eval(0.0f);
	//sigmaPhi1 = target.sigma.phi.eval(1.0f);

	//// Orginal paper synthesis.
	//for (int c = 0; c < 3; ++c) {
	//	target.Kd[c].factors[0] = (Kd0[c] - Kd1[c]) / (KdPhi0[c] - KdPhi1[c]);
	//	target.Kd[c].factors[3] = (KdPhi0[c] * Kd1[c] - KdPhi1[c] * Kd0[c]) / (KdPhi0[c] - KdPhi1[c]);
	//	cv::resize(source.Kd[c].factors[1], target.Kd[c].factors[1], cv::Size(targetHeight, targetWidth));
	//	cv::resize(source.Kd[c].factors[2], target.Kd[c].factors[2], cv::Size(targetHeight, targetWidth));
	//}
	//target.Ks.factors[0] = (Ks0 - Ks1) / (KsPhi0 - KsPhi1);
	//target.Ks.factors[3] = (KsPhi0 * Ks1 - KsPhi1 * Ks0) / (KsPhi0 - KsPhi1);
	//cv::resize(source.Ks.factors[1], target.Ks.factors[1], cv::Size(targetHeight, targetWidth));
	//cv::resize(source.Ks.factors[2], target.Ks.factors[2], cv::Size(targetHeight, targetWidth));
	//target.sigma.factors[0] = (sigma0 - sigma1) / (sigmaPhi0 - sigmaPhi1);
	//target.sigma.factors[3] = (sigmaPhi0 * sigma1 - sigmaPhi1 * Ks0) / (sigmaPhi0 - sigmaPhi1);
	//cv::resize(source.sigma.factors[1], target.sigma.factors[1], cv::Size(targetHeight, targetWidth));
	//cv::resize(source.sigma.factors[2], target.sigma.factors[2], cv::Size(targetHeight, targetWidth));

	//// Export.
	//target.exportFrames(outFilepath + "/spatial-ref/images");
	//target.save(outFilepath + "/spatial-ref/staf");

}

void temporalPrediction(const std::string & srcFilepath, const std::string & tgtFilename, const std::string & outFilepath, float t0 = 0.0f) {

	// Source.
  PolyTSVBRDF source(srcFilepath);

	// Source style channels.
  int numStyleChannels = 5 * (Parameter::DEGREE + 1);
	std::vector<cv::Mat> sourceChannels;
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.Kd[c].coefs[i]);
  for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.Ks.coefs[i]);
  for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.sigma.coefs[i]);
	cv::Mat sourceStyles;
	cv::merge(sourceChannels, sourceStyles);

	// Source guide channels.
	int numGuideChannels = 1;
	cv::Mat sourceGuides = 0.2125f * source.getKd(t0, 0) + 0.7154f * source.getKd(t0, 1) + 0.0721f * source.getKd(t0, 2);
	sourceGuides.convertTo(sourceGuides, CV_8U, 255.0f);
	equalizeHist(sourceGuides, sourceGuides);
	sourceGuides.convertTo(sourceGuides, CV_32F, 1.0f / 255.0f);

	// Target.
	cv::Mat target = cv::imread(tgtFilename, CV_LOAD_IMAGE_UNCHANGED);
	//cv::resize(target, target, cv::Size(300,300));
	cv::Mat targetKd[3];
	for (int c = 0; c < 3; ++c) {
		cv::extractChannel(target, targetKd[c], c);
		targetKd[c].convertTo(targetKd[c], CV_32F, 1.0f / 255.0f);
	}
	int targetHeight = target.size[0];
	int targetWidth = target.size[1];

	// Target style channels => output.
	cv::Mat targetStyles(target.size[0], targetWidth, sourceStyles.type());

	// Target guide channels.
	cv::Mat targetGuides = 0.2125f * targetKd[0] + 0.7154f * targetKd[1] + 0.0721f * targetKd[2];
	targetGuides.convertTo(targetGuides, CV_8U, 255.0f);
	equalizeHist(targetGuides, targetGuides);
	targetGuides.convertTo(targetGuides, CV_32F, 1.0f / 255.0f);

	// Style weights.
	const float totalStyleWeight = 2.0f;
	std::vector<float> styleWeights(numStyleChannels);
	for (int i = 0; i < 5 * (Parameter::DEGREE + 1); ++i) {
		if (i < 4 * (Parameter::DEGREE + 1)) styleWeights[i] = totalStyleWeight / numStyleChannels;
		else styleWeights[i] = 0.0f;
	}
	
  // Guide weights.
	const float totalGuideWeight = 1.0f;
	std::vector<float> guideWeights(numGuideChannels);
	for (int i = 0; i < numGuideChannels; i++)
		guideWeights[i] = totalGuideWeight / numGuideChannels;

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

	// EBSynth.
	ebsynthRun(
		EBSYNTH_BACKEND_CUDA,
		numStyleChannels,
		numGuideChannels,
		source.width,
		source.height,
		sourceStyles.data,
		sourceGuides.data,
		targetWidth,
		targetHeight,
		targetGuides.data,
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
    Parameter::DEGREE,
		targetStyles.data
	);

	// Output.
  PolyTSVBRDF reconstruct;
	reconstruct.resize(targetWidth, targetHeight, source.type());

	// Split channels.
	std::vector<cv::Mat> reconstructChannels;
	for (int c = 0; c < 3; ++c)
		for (int i = 0; i <= Parameter::DEGREE; ++i) reconstructChannels.push_back(reconstruct.Kd[c].coefs[i]);
	for (int i = 0; i <= Parameter::DEGREE; ++i) reconstructChannels.push_back(reconstruct.Ks.coefs[i]);
	for (int i = 0; i <= Parameter::DEGREE; ++i) reconstructChannels.push_back(reconstruct.sigma.coefs[i]);
	cv::split(targetStyles, reconstructChannels);

	// Export.
	reconstruct.exportFrames(outFilepath + "/temporal/images");
	reconstruct.save(outFilepath + "/temporal/staf");

	// Correction.
	for (int c = 0; c < 3; ++c) {
		cv::Mat recKd = reconstruct.getKd(t0, c);
    for (int i = 0; i <= Parameter::DEGREE; ++i)
      reconstruct.Kd[c].coefs[i] = (targetKd[c].mul(1.0f / recKd)).mul(reconstruct.Kd[c].coefs[i]);
	}

	// Export.
	reconstruct.exportFrames(outFilepath + "/transfer/images");
	reconstruct.save(outFilepath + "/transfer/staf");

}

void temporalPredictionRef(const std::string & srcFilepath, const std::string & tgtFilename, const std::string & outFilepath, float t0 = 0.0f) {

	// Source.
  STAFTSVBRDF source(srcFilepath);

	// Target.
	cv::Mat target = cv::imread(tgtFilename, CV_LOAD_IMAGE_UNCHANGED);
	cv::resize(target, target, cv::Size(source.height, source.width));
	cv::Mat targetKd[3];
	for (int c = 0; c < 3; ++c) {
		cv::extractChannel(target, targetKd[c], c);
		targetKd[c].convertTo(targetKd[c], CV_32F, 1.0f / 255.0f);
	}

	// Output.
  STAFTSVBRDF reconstruct;
	reconstruct.resize(source.width, source.height, source.Kd[0].factors[0].type());

	// Reconstruct.
	reconstruct.Kd[0].phi = source.Kd[0].phi;
	reconstruct.Kd[1].phi = source.Kd[1].phi;
	reconstruct.Kd[2].phi = source.Kd[2].phi;
	reconstruct.Ks.phi = source.Ks.phi;
	reconstruct.sigma.phi = source.sigma.phi;
	for (int f = 0; f < 4; ++f) {
		for (int c = 0; c < 3; ++c)
			reconstruct.Kd[c].factors[f] = source.Kd[c].factors[f];
		reconstruct.Ks.factors[f] = source.Ks.factors[f];
		reconstruct.sigma.factors[f] = source.sigma.factors[f];
	}

	// Original paper transfer.
	for (int c = 0; c < 3; ++c) {
		cv::Mat recKd = reconstruct.getKd(t0, c);
		reconstruct.Kd[c].factors[0] = (targetKd[c].mul(1.0f / recKd)).mul(reconstruct.Kd[c].factors[0]);
		reconstruct.Kd[c].factors[3] = (targetKd[c].mul(1.0f / recKd)).mul(reconstruct.Kd[c].factors[3]);
	}

	// Export.
	reconstruct.exportFrames(outFilepath + "/transfer-ref/images");
	reconstruct.save(outFilepath + "/transfer-ref/staf");

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
		//spatialPredictionRef(argv[1], argv[2]);
	}
	
	else if (argc == 5) {
		temporalPrediction(argv[1], argv[2], argv[3], std::stof(argv[4]));
		temporalPredictionRef(argv[1], argv[2], argv[3], std::stof(argv[4]));
	}

	return 0;
}
