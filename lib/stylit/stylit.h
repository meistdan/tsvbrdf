#ifndef LIBSTYLIT_STYLIT_H
#define LIBSTYLIT_STYLIT_H

#ifndef LIBSTYLIT_API
  #ifdef WIN32
    #define LIBSTYLIT_API __declspec(dllimport)
  #else
    #define LIBSTYLIT_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define STYLIT_BACKEND_CPU         0x0001
#define STYLIT_BACKEND_CUDA        0x0002

#define STYLIT_NUM_STYLE_CHANNELS  20
#define STYLIT_MAX_GUIDE_CHANNELS  24

#define STYLIT_VOTEMODE_PLAIN      0x0001         // weight = 1
#define STYLIT_VOTEMODE_WEIGHTED   0x0002         // weight = 1/(1+error)

LIBSTYLIT_API
int stylitBackendAvailable(int stylitBackend);    // returns non-zero if the specified backend is available

LIBSTYLIT_API
void stylitRun(int    stylitBackend,              // use BACKEND_CUDA for maximum speed, BACKEND_CPU for compatibility

               int    numStyleChannels,           // must be equal to STYLIT_NUM_STYLE_CHANNELS
               int    numGuideChannels,

               int    sourceWidth,
               int    sourceHeight,
               void*  sourceStyleData,            // (width * height * numStyleChannels) floats, scan-line order
               void*  sourceGuideData,            // (width * height * numGuideChannels) floats, scan-line order

               int    targetWidth,
               int    targetHeight,
               void*  targetGuideData,            // (width * height * numGuideChannels) floats, scan-line order
               void*  targetModulationData,       // (width * height * numGuideChannels) bytes, scan-line order; pass NULL to switch off the modulation

               float* styleWeights,               // (numStyleChannels) floats
               float* guideWeights,               // (numGuideChannels) floats

                                                  // guideError(txy,sxy,ch) = guideWeights[ch] * (targetModulation[txy][ch]/255) * (targetGuide[txy][ch]-sourceGuide[sxy][ch])^2

               float  uniformityWeight,           // reasonable values are between 500-15000, 3500 is a good default

               int    patchSize,                  // odd sizes only, use 5 for 5x5 patch, 7 for 7x7, etc.
               int    voteMode,                   // use VOTEMODE_WEIGHTED for sharper result

               int    numPyramidLevels,

               int*   numSearchVoteItersPerLevel, // how many search/vote iters to perform at each level (array of ints, coarse first, fine last)
               int*   numPatchMatchItersPerLevel, // how many Patch-Match iters to perform at each level (array of ints, coarse first, fine last)

               int*   stopThresholdPerLevel,      // stop improving pixel when its change since last iteration falls under this threshold

               int    extraPass3x3,               // !=0 performs additional polishing pass with 3x3 patches at the finest level

               int    liveUpdate,                 // !=0 progressively updates the output buffer with intermediate solution after each iteration

               void*  outputData                  // (width * height * numStyleChannels) floats, scan-line order
              );


#ifdef __cplusplus
}
#endif

#endif
