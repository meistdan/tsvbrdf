#include "TSVBRDF.h"
#include <chrono>

#define CORRECTION 0

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

void exportFrames(const std::string & outFilepath) {
#if 1
  PolyTSVBRDF source(outFilepath);
  source.exportFrames(outFilepath + "/images");
#else 
  const int FRAMES = 51;
  std::vector<cv::Mat> data;
  cv::Mat img;
  for (int i = 0; i < FRAMES; ++i) {
      img = cv::imread(outFilepath + "/Diffuse-" + std::to_string(i) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
      data.push_back(img);
  }
  const int OUT_FRAMES = 21;
  for (int i = 0; i < OUT_FRAMES; ++i) {
      float t = float(i) / (OUT_FRAMES - 1);
      float j = t * (FRAMES - 1);
      int j0 = int(j);
      int j1 = std::ceil(j);
      float alpha = j - j0;
      if (j0 == j1) img = data[j0];
      else img = (1.0 - alpha) * data[j0] + alpha * data[j1];
      pow(img, 0.5f, img);
      img = 255.0f * img;
      imwrite(outFilepath + "/images/" + std::to_string(i) + ".jpg", img);
  }
#endif
}

void spatialPrediction(const std::string & srcFilepath, const std::string & outFilepath) {

  // Source.
  PolyTSVBRDF source(srcFilepath);

  // Source guide channels.
  int sn = source.width * source.height;
  int numGuideChannels = 1;
  std::vector<float> sourceGuides(numGuideChannels * sn);

  // Source style channels.
  int numStyleChannels = 7 * (Parameter::DEGREE + 1);
  std::vector<cv::Mat> sourceChannels;
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.diffuse[c].coefs[i]);
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.specular[c].coefs[i]);
  for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.roughness.coefs[i]);
  cv::Mat sourceStyles;
  cv::merge(sourceChannels, sourceStyles);

  // Target resolution.
  const int SIZE_MULT = 2;
  int targetWidth = SIZE_MULT * source.width;
  int targetHeight = SIZE_MULT * source.height;

  // Target guide channels.
  int tn = targetWidth * targetHeight;
  std::vector<float> targetGuides(numGuideChannels * tn);

  // Target style channels => output.
  cv::Mat targetStyles(targetHeight, targetWidth, sourceStyles.type());

  // Style weights.
  std::vector<float> styleWeights(numStyleChannels);
  for (int i = 0; i < 7 * (Parameter::DEGREE + 1); ++i)
      if (i < 3 * (Parameter::DEGREE + 1))
          styleWeights[i] = 1.0f;
      else if (i < 6 * (Parameter::DEGREE + 1))
          styleWeights[i] = 1.0f;
      else
          styleWeights[i] = 0.0f;

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
  auto t1 = std::chrono::high_resolution_clock::now();
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
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
  std::cout << "Synthesis done in " << time << "s" << std::endl;

  // Output.
  PolyTSVBRDF target(targetWidth, targetHeight, source.type());

  // Split channels.
  std::vector<cv::Mat> targetChannels;
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i <= Parameter::DEGREE; ++i) targetChannels.push_back(target.diffuse[c].coefs[i]);
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i <= Parameter::DEGREE; ++i) targetChannels.push_back(target.specular[c].coefs[i]);
  for (int i = 0; i <= Parameter::DEGREE; ++i) targetChannels.push_back(target.roughness.coefs[i]);
  cv::split(targetStyles, targetChannels);

  // Export.
  source.exportFrames(srcFilepath + "/images");
  target.exportFrames(outFilepath + "/images");
  target.save(outFilepath);

}

void temporalPrediction(const std::string & srcFilepath, const std::string & tgtFilename, const std::string & outFilepath, float t0 = 0.0f) {

  // Source.
  PolyTSVBRDF source(srcFilepath);

  // Source style channels.
  int numStyleChannels = 7 * (Parameter::DEGREE + 1);
  std::vector<cv::Mat> sourceChannels;
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.diffuse[c].coefs[i]);
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.specular[c].coefs[i]);
  for (int i = 0; i <= Parameter::DEGREE; ++i) sourceChannels.push_back(source.roughness.coefs[i]);
  cv::Mat sourceStyles;
  cv::merge(sourceChannels, sourceStyles);

  // Source guide channels.
  int numGuideChannels = 1;
  cv::Mat sourceGuides = 0.2125f * source.getDiffuse(t0, 0) + 0.7154f * source.getDiffuse(t0, 1) + 0.0721f * source.getDiffuse(t0, 2);
  sourceGuides.convertTo(sourceGuides, CV_8U, 255.0f);
  equalizeHist(sourceGuides, sourceGuides);
  sourceGuides.convertTo(sourceGuides, CV_32F, 1.0f / 255.0f);

  // Target.
  cv::Mat target = cv::imread(tgtFilename, CV_LOAD_IMAGE_UNCHANGED);
  //cv::resize(target, target, cv::Size(220, 220));
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
  const float totalStyleWeight = 1.0f;
  std::vector<float> styleWeights(numStyleChannels);
  for (int i = 0; i < numStyleChannels; ++i)
    //styleWeights[i] = totalStyleWeight / numStyleChannels;
    styleWeights[i] = 1.0f;

  // Guide weights.
  const float totalGuideWeight = 1.0f;
  std::vector<float> guideWeights(numGuideChannels);
  for (int i = 0; i < numGuideChannels; i++)
    //guideWeights[i] = totalGuideWeight / numGuideChannels;
    guideWeights[i] = 1.0f;

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
  auto t1 = std::chrono::high_resolution_clock::now();
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
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
  std::cout << "Synthesis done in " << time << "s" << std::endl;

  // Output.
  PolyTSVBRDF reconstruct;
  reconstruct.resize(targetWidth, targetHeight, source.type());

  // Split channels.
  std::vector<cv::Mat> reconstructChannels;
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i <= Parameter::DEGREE; ++i) reconstructChannels.push_back(reconstruct.diffuse[c].coefs[i]);
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i <= Parameter::DEGREE; ++i) reconstructChannels.push_back(reconstruct.specular[c].coefs[i]);
  for (int i = 0; i <= Parameter::DEGREE; ++i) reconstructChannels.push_back(reconstruct.roughness.coefs[i]);
  cv::split(targetStyles, reconstructChannels);

#if CORRECTION
  // Correction.
  for (int c = 0; c < 3; ++c) {
    cv::Mat recKd = reconstruct.getDiffuse(t0, c);
    for (int i = 0; i <= Parameter::DEGREE; ++i)
      reconstruct.diffuse[c].coefs[i] = (targetKd[c].mul(1.0f / recKd)).mul(reconstruct.diffuse[c].coefs[i]);
  }
#endif

  // Export.
  reconstruct.exportFrames(outFilepath + "/images");
  reconstruct.save(outFilepath);

}

void temporalPredictionRef(const std::string & srcFilepath, const std::string & tgtFilename, const std::string & outFilepath, float t0 = 0.0f) {

    // Source.
    PolyTSVBRDF source(srcFilepath);
    PolyTSVBRDF reconstruct(srcFilepath);

    // Target.
    cv::Mat target = cv::imread(tgtFilename, CV_LOAD_IMAGE_UNCHANGED);
    cv::resize(target, target, cv::Size(220, 220));
    cv::Mat targetKd[3];
    for (int c = 0; c < 3; ++c) {
        cv::extractChannel(target, targetKd[c], c);
        targetKd[c].convertTo(targetKd[c], CV_32F, 1.0f / 255.0f);
    }
    int targetHeight = target.size[0];
    int targetWidth = target.size[1];

    // Correction.
    for (int c = 0; c < 3; ++c) {
        cv::Mat recKd = reconstruct.getDiffuse(t0, c);
        for (int i = 0; i <= Parameter::DEGREE; ++i)
            reconstruct.diffuse[c].coefs[i] = (targetKd[c].mul(1.0f / recKd)).mul(reconstruct.diffuse[c].coefs[i]);
    }

    // Export.
    reconstruct.exportFrames(outFilepath + "/images");
    reconstruct.save(outFilepath);

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
    exportFrames(argv[1]);
  } 
  else if (argc == 3) {
    spatialPrediction(argv[1], argv[2]);
  }
  else if (argc == 5) {
    temporalPrediction(argv[1], argv[2], argv[3], std::stof(argv[4]));
  }

  return 0;
}
