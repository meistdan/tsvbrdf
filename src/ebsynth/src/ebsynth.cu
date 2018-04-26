// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#include "ebsynth.h"
#include "patchmatch_gpu.h"

#define FOR(A,X,Y) for(int Y=0;Y<A.height();Y++) for(int X=0;X<A.width();X++)

A2V2i nnfInitRandom(const V2i& targetSize,
                    const V2i& sourceSize,
                    const int  patchSize)
{
  A2V2i NNF(targetSize);
  const int r = patchSize/2;

  for (int i = 0; i < NNF.numel(); i++)
  {
      NNF[i] = V2i
      (
          r+(rand()%(sourceSize[0]-2*r)),
          r+(rand()%(sourceSize[1]-2*r))
      );
  }

  return NNF;
}

A2V2i nnfUpscale(const A2V2i& NNF,
                 const int    patchSize,
                 const V2i&   targetSize,
                 const V2i&   sourceSize)
{
  A2V2i NNF2x(targetSize);

  FOR(NNF2x,x,y)
  {
    NNF2x(x,y) = NNF(clamp(x/2,0,NNF.width()-1),
                     clamp(y/2,0,NNF.height()-1))*2+V2i(x%2,y%2);
  }

  FOR(NNF2x,x,y)
  {
    const V2i nn = NNF2x(x,y);

    NNF2x(x,y) = V2i(clamp(nn(0),patchSize,sourceSize(0)-patchSize-1),
                     clamp(nn(1),patchSize,sourceSize(1)-patchSize-1));
  }

  return NNF2x;
}

template<int N, typename T, int M>
__global__ void krnlVotePlain(      TexArray2<N,T,M> target,
                              const TexArray2<N,T,M> source,
                              const TexArray2<2,int> NNF,
                              const int              patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<target.width && y<target.height)
  {
    const int r = patchSize / 2;

    Vec<N,float> sumColor = zero<Vec<N,float>>::value();
    float sumWeight = 0;

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      /*
      if
      (
        x+px >= 0 && x+px < NNF.width () &&
        y+py >= 0 && y+py < NNF.height()
      )
      */
      {
        const V2i n = NNF(x+px,y+py)-V2i(px,py);

        /*if
        (
          n[0] >= 0 && n[0] < S.width () &&
          n[1] >= 0 && n[1] < S.height()
        )*/
        {
          const float weight = 1.0f;
          sumColor += weight*Vec<N,float>(source(n(0),n(1)));
          sumWeight += weight;
        }
      }
    }

    const Vec<N,T> v = Vec<N,T>(sumColor/sumWeight);
    target.write(x,y,v);
  }
}

template<int N, typename T, int M>
__global__ void krnlVoteWeighted(      TexArray2<N,T,M>   target,
                                 const TexArray2<N,T,M>   source,
                                 const TexArray2<2,int>   NNF,
                                 const TexArray2<1,float> E,
                                 const int patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<target.width && y<target.height)
  {
    const int r = patchSize / 2;

    Vec<N,float> sumColor = zero<Vec<N,float>>::value();
    float sumWeight = 0;

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      /*
      if
      (
        x+px >= 0 && x+px < NNF.width () &&
        y+py >= 0 && y+py < NNF.height()
      )
      */
      {
        const V2i n = NNF(x+px,y+py)-V2i(px,py);

        /*if
        (
          n[0] >= 0 && n[0] < S.width () &&
          n[1] >= 0 && n[1] < S.height()
        )*/
        {
          const float error = E(x+px,y+py)(0)/(patchSize*patchSize*N);
          const float weight = 1.0f/(1.0f+error);
          sumColor += weight*Vec<N,float>(source(n(0),n(1)));
          sumWeight += weight;
        }
      }
    }

    const Vec<N,T> v = Vec<N,T>(sumColor/sumWeight);
    target.write(x,y,v);
  }
}

template<int N, typename T, int M>
__device__ Vec<N,T> sampleBilinear(const TexArray2<N,T,M>& I,float x,float y)
{
  const int ix = x;
  const int iy = y;

  const float s = x-ix;
  const float t = y-iy;

  // XXX: clamp!!!
  return Vec<N,T>((1.0f-s)*(1.0f-t)*Vec<N,float>(I(ix  ,iy  ))+
                  (     s)*(1.0f-t)*Vec<N,float>(I(ix+1,iy  ))+
                  (1.0f-s)*(     t)*Vec<N,float>(I(ix  ,iy+1))+
                  (     s)*(     t)*Vec<N,float>(I(ix+1,iy+1)));
};

template<int N, typename T, int M>
__global__ void krnlResampleBilinear(TexArray2<N,T,M> O,
                                     const TexArray2<N,T,M> I)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<O.width && y<O.height)
  {
    const float s = float(I.width)/float(O.width);
    O.write(x,y,sampleBilinear(I,s*float(x),s*float(y)));
  }
}

template<int N, typename T, int M>
__global__ void krnlEvalMask(      TexArray2<1,unsigned char> mask,
                             const TexArray2<N,T,M> style,
                             const TexArray2<N,T,M> style2,
                             const int stopThreshold)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<mask.width && y<mask.height)
  {
    const Vec<N,T> s  = style(x,y);
    const Vec<N,T> s2 = style2(x,y);

    int maxDiff = 0;
    for(int c=0;c<N;c++)
    {
      const int diff = std::abs(int(s[c])-int(s2[c]));
      maxDiff = diff>maxDiff ? diff:maxDiff;
    }

    const Vec<1,unsigned char> msk = maxDiff < stopThreshold ? Vec<1,unsigned char>(0) : Vec<1,unsigned char>(255);

    mask.write(x,y,msk);
  }
}

__global__ void krnlDilateMask(TexArray2<1,unsigned char> mask2,
                               const TexArray2<1,unsigned char> mask,
                               const int patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<mask.width && y<mask.height)
  {
    const int r = patchSize / 2;

    Vec<1,unsigned char> msk = Vec<1,unsigned char>(0);

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      if (mask(x+px,y+py)[0]==255) { msk = Vec<1,unsigned char>(255); }
    }

    mask2.write(x,y,msk);
  }
}

template<int N, typename T, int M>
void resampleGPU(      TexArray2<N,T,M>& O,
                 const TexArray2<N,T,M>& I)
{
  const int numThreadsPerBlock = 24;
  const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
  const dim3 numBlocks = dim3((O.width+threadsPerBlock.x)/threadsPerBlock.x,
                              (O.height+threadsPerBlock.y)/threadsPerBlock.y);

  krnlResampleBilinear<<<numBlocks,threadsPerBlock>>>(O,I);

  checkCudaError(cudaDeviceSynchronize());
}

template<int NS,int NG,typename T>
struct PatchSSD_Split
{
  const TexArray2<NS,T> targetStyle;
  const TexArray2<NS,T> sourceStyle;

  const TexArray2<NG,T> targetGuide;
  const TexArray2<NG,T> sourceGuide;

  const Vec<NS,float> styleWeights;
  const Vec<NG,float> guideWeights;

  PatchSSD_Split(const TexArray2<NS,T>& targetStyle,
                 const TexArray2<NS,T>& sourceStyle,

                 const TexArray2<NG,T>& targetGuide,
                 const TexArray2<NG,T>& sourceGuide,

                 const Vec<NS,float>&   styleWeights,
                 const Vec<NG,float>&   guideWeights)

  : targetStyle(targetStyle),sourceStyle(sourceStyle),
    targetGuide(targetGuide),sourceGuide(sourceGuide),
    styleWeights(styleWeights),guideWeights(guideWeights) {}

   __device__ float operator()(const int   patchSize,
                               const int   tx,
                               const int   ty,
                               const int   sx,
                               const int   sy,
                               const float ebest)
  {
    const int r = patchSize/2;
    float error = 0;

    for(int py=-r;py<=+r;py++)
    {
      for(int px=-r;px<=+r;px++)
      {
        {
          const Vec<NS,T> pixTs = targetStyle(tx + px,ty + py);
          const Vec<NS,T> pixSs = sourceStyle(sx + px,sy + py);
          for(int i=0;i<NS;i++)
          {
            const float diff = float(pixTs[i]) - float(pixSs[i]);
            error += styleWeights[i]*diff*diff;
          }
        }

        {
          const Vec<NG,T> pixTg = targetGuide(tx + px,ty + py);
          const Vec<NG,T> pixSg = sourceGuide(sx + px,sy + py);
          for(int i=0;i<NG;i++)
          {
            const float diff = float(pixTg[i]) - float(pixSg[i]);
            error += guideWeights[i]*diff*diff;
          }
        }
      }

      if (error>ebest) { return error; }
    }

    return error;
  }
};

template<int NS,int NG,typename T>
struct PatchSSD_Split_Modulation
{
  const TexArray2<NS,T> targetStyle;
  const TexArray2<NS,T> sourceStyle;

  const TexArray2<NG,T> targetGuide;
  const TexArray2<NG,T> sourceGuide;

  const TexArray2<NG,unsigned char> targetModulation;

  const Vec<NS,float> styleWeights;
  const Vec<NG,float> guideWeights;

  PatchSSD_Split_Modulation(const TexArray2<NS,T>& targetStyle,
                            const TexArray2<NS,T>& sourceStyle,

                            const TexArray2<NG,T>& targetGuide,
                            const TexArray2<NG,T>& sourceGuide,

                            const TexArray2<NG,unsigned char>& targetModulation,

                            const Vec<NS,float>&   styleWeights,
                            const Vec<NG,float>&   guideWeights)

  : targetStyle(targetStyle),sourceStyle(sourceStyle),
    targetGuide(targetGuide),sourceGuide(sourceGuide),
    targetModulation(targetModulation),
    styleWeights(styleWeights),guideWeights(guideWeights) {}

   __device__ float operator()(const int   patchSize,
                               const int   tx,
                               const int   ty,
                               const int   sx,
                               const int   sy,
                               const float ebest)
  {
    const int r = patchSize/2;
    float error = 0;

    for(int py=-r;py<=+r;py++)
    {
      for(int px=-r;px<=+r;px++)
      {
        {
          const Vec<NS,T> pixTs = targetStyle(tx + px,ty + py);
          const Vec<NS,T> pixSs = sourceStyle(sx + px,sy + py);
          for(int i=0;i<NS;i++)
          {
            const float diff = float(pixTs[i]) - float(pixSs[i]);
            error += styleWeights[i]*diff*diff;
          }
        }

        {
          const Vec<NG,T> pixTg = targetGuide(tx + px,ty + py);
          const Vec<NG,T> pixSg = sourceGuide(sx + px,sy + py);
          const Vec<NG,float> mult = Vec<NG,float>(targetModulation(tx,ty))/255.0f;

          for(int i=0;i<NG;i++)
          {
            const float diff = float(pixTg[i]) - float(pixSg[i]);
            error += guideWeights[i]*mult[i]*diff*diff;
          }
        }
      }

      if (error>ebest) { return error; }
    }

    return error;
  }
};

V2i pyramidLevelSize(const V2i& sizeBase,const int numLevels,const int level)
{
  return V2i(V2f(sizeBase)*pow(2.0f,-float(numLevels-1-level)));
}

template<int NS,int NG>
void runEbsynth(int    ebsynthBackend,
                int    numStyleChannels,
                int    numGuideChannels,
                int    sourceWidth,
                int    sourceHeight,
                void*  sourceStyleData,
                void*  sourceGuideData,
                int    targetWidth,
                int    targetHeight,
                void*  targetGuideData,
                void*  targetModulationData,
                float* styleWeights,
                float* guideWeights,
                float  uniformityWeight,
                int    patchSize,
                int    voteMode,
                int    numPyramidLevels,
                int*   numSearchVoteItersPerLevel,
                int*   numPatchMatchItersPerLevel,
                int*   stopThresholdPerLevel,
                void*  outputData)
{
  const int levelCount = numPyramidLevels;

  struct PyramidLevel
  {
    PyramidLevel() { }

    int sourceWidth;
    int sourceHeight;
    int targetWidth;
    int targetHeight;

    TexArray2<NS,float> sourceStyle;
    TexArray2<NG,float> sourceGuide;
    TexArray2<NS,float> targetStyle;
    TexArray2<NS,float> targetStyle2;
    TexArray2<1,unsigned char>  mask;
    TexArray2<1,unsigned char>  mask2;
    TexArray2<NG,float> targetGuide;
    TexArray2<NG,unsigned char> targetModulation;
    TexArray2<2,int>            NNF;
    TexArray2<2,int>            NNF2;
    TexArray2<1,float>          E;
    MemArray2<int>              Omega;
  };

  std::vector<PyramidLevel> pyramid(levelCount);
  for(int level=0;level<levelCount;level++)
  {
    const V2i levelSourceSize = pyramidLevelSize(V2i(sourceWidth,sourceHeight),levelCount,level);
    const V2i levelTargetSize = pyramidLevelSize(V2i(targetWidth,targetHeight),levelCount,level);

    pyramid[level].sourceWidth  = levelSourceSize(0);
    pyramid[level].sourceHeight = levelSourceSize(1);
    pyramid[level].targetWidth  = levelTargetSize(0);
    pyramid[level].targetHeight = levelTargetSize(1);

    pyramid[level].sourceStyle  = TexArray2<NS,float>(levelSourceSize);
    pyramid[level].sourceGuide  = TexArray2<NG,float>(levelSourceSize);
    pyramid[level].targetStyle  = TexArray2<NS,float>(levelTargetSize);
    pyramid[level].targetStyle2 = TexArray2<NS,float>(levelTargetSize);
    pyramid[level].mask         = TexArray2<1,unsigned char>(levelTargetSize);
    pyramid[level].mask2        = TexArray2<1,unsigned char>(levelTargetSize);
    pyramid[level].targetGuide  = TexArray2<NG,float>(levelTargetSize);
    pyramid[level].NNF          = TexArray2<2,int>  (levelTargetSize);
    pyramid[level].NNF2         = TexArray2<2,int>  (levelTargetSize);
    pyramid[level].E            = TexArray2<1,float>(levelTargetSize);
    pyramid[level].Omega        = MemArray2<int>    (levelSourceSize);

    if (targetModulationData) { pyramid[level].targetModulation = TexArray2<NG,unsigned char>(levelTargetSize); }
  }

  copy(&pyramid[levelCount-1].sourceStyle,sourceStyleData);
  copy(&pyramid[levelCount-1].sourceGuide,sourceGuideData);
  copy(&pyramid[levelCount-1].targetGuide,targetGuideData);
  if (targetModulationData) { copy(&pyramid[levelCount-1].targetModulation,targetModulationData); }

  for(int level=0;level<levelCount-1;level++)
  {
    resampleGPU(pyramid[level].sourceStyle,pyramid[levelCount-1].sourceStyle);
    resampleGPU(pyramid[level].sourceGuide,pyramid[levelCount-1].sourceGuide);
    resampleGPU(pyramid[level].targetGuide,pyramid[levelCount-1].targetGuide);
    if (targetModulationData) { resampleGPU(pyramid[level].targetModulation,pyramid[levelCount-1].targetModulation); }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  bool inExtraPass = false;

  curandState* rngStates = initGpuRng(targetWidth,targetHeight);

  for (int level=0;level<pyramid.size();level++)
  {
    /////////////////////////////////////////////////////////////////////////////

    if (!inExtraPass)
    {
      A2V2i cpu_NNF;
      if (level>0)
      {
        A2V2i prevLevelNNF(pyramid[level-1].targetWidth,
                           pyramid[level-1].targetHeight);

        copy(&prevLevelNNF,pyramid[level-1].NNF);

        cpu_NNF = nnfUpscale(prevLevelNNF,
                             patchSize,
                             V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                             V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight));
      }
      else
      {
        cpu_NNF = nnfInitRandom(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                                V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                                patchSize);
      }
      copy(&pyramid[level].NNF,cpu_NNF);

      /////////////////////////////////////////////////////////////////////////
      Array2<int> cpu_Omega(pyramid[level].sourceWidth,pyramid[level].sourceHeight);

      fill(&cpu_Omega,(int)0);
      for(int ay=0;ay<cpu_NNF.height();ay++)
      for(int ax=0;ax<cpu_NNF.width();ax++)
      {
        const V2i& n = cpu_NNF(ax,ay);
        const int bx = n(0);
        const int by = n(1);

        const int r = patchSize/2;

        for(int oy=-r;oy<=+r;oy++)
        for(int ox=-r;ox<=+r;ox++)
        {
          const int x = bx+ox;
          const int y = by+oy;
          cpu_Omega(x,y) += 1;
        }
      }

      copy(&pyramid[level].Omega,cpu_Omega);
      /////////////////////////////////////////////////////////////////////////
    }

    ////////////////////////////////////////////////////////////////////////////
    {
      const int numThreadsPerBlock = 24;
      const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
      const dim3 numBlocks = dim3((pyramid[level].targetWidth+threadsPerBlock.x)/threadsPerBlock.x,
                                  (pyramid[level].targetHeight+threadsPerBlock.y)/threadsPerBlock.y);

      krnlVotePlain<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                   pyramid[level].sourceStyle,
                                                   pyramid[level].NNF,
                                                   patchSize);

      std::swap(pyramid[level].targetStyle2,pyramid[level].targetStyle);
      checkCudaError( cudaDeviceSynchronize() );
    }
    ////////////////////////////////////////////////////////////////////////////

    Array2<Vec<1,unsigned char>> cpu_mask(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight));
    fill(&cpu_mask,Vec<1,unsigned char>(255));
    copy(&pyramid[level].mask,cpu_mask);

    ////////////////////////////////////////////////////////////////////////////

    for (int voteIter=0;voteIter<numSearchVoteItersPerLevel[level];voteIter++)
    {
      Vec<NS,float> styleWeightsVec;
      for(int i=0;i<NS;i++) { styleWeightsVec[i] = styleWeights[i]; }

      Vec<NG,float> guideWeightsVec;
      for(int i=0;i<NG;i++) { guideWeightsVec[i] = guideWeights[i]; }

      const int numGpuThreadsPerBlock = 24;

      if (numPatchMatchItersPerLevel[level]>0)
      {
        if (targetModulationData)
        {
          patchmatchGPU(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                        V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                        pyramid[level].Omega,
                        patchSize,
                        PatchSSD_Split_Modulation<NS,NG,float>(pyramid[level].targetStyle,
                                                                       pyramid[level].sourceStyle,
                                                                       pyramid[level].targetGuide,
                                                                       pyramid[level].sourceGuide,
                                                                       pyramid[level].targetModulation,
                                                                       styleWeightsVec,
                                                                       guideWeightsVec),
                        uniformityWeight,
                        numPatchMatchItersPerLevel[level],
                        numGpuThreadsPerBlock,
                        pyramid[level].NNF,
                        pyramid[level].NNF2,
                        pyramid[level].E,
                        pyramid[level].mask,
                        rngStates);
        }
        else
        {
          patchmatchGPU(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                        V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                        pyramid[level].Omega,
                        patchSize,
                        PatchSSD_Split<NS,NG,float>(pyramid[level].targetStyle,
                                                            pyramid[level].sourceStyle,
                                                            pyramid[level].targetGuide,
                                                            pyramid[level].sourceGuide,
                                                            styleWeightsVec,
                                                            guideWeightsVec),
                        uniformityWeight,
                        numPatchMatchItersPerLevel[level],
                        numGpuThreadsPerBlock,
                        pyramid[level].NNF,
                        pyramid[level].NNF2,
                        pyramid[level].E,
                        pyramid[level].mask,
                        rngStates);
        }
      }
      else
      {
        const int numThreadsPerBlock = 24;
        const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
        const dim3 numBlocks = dim3((pyramid[level].targetWidth+threadsPerBlock.x)/threadsPerBlock.x,
                                    (pyramid[level].targetHeight+threadsPerBlock.y)/threadsPerBlock.y);

        if (targetModulationData)
        {
          krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchSize,
                                                           PatchSSD_Split_Modulation<NS,NG,float>(pyramid[level].targetStyle,
                                                                                                  pyramid[level].sourceStyle,
                                                                                                  pyramid[level].targetGuide,
                                                                                                  pyramid[level].sourceGuide,
                                                                                                  pyramid[level].targetModulation,
                                                                                                  styleWeightsVec,
                                                                                                  guideWeightsVec),
                                                           pyramid[level].NNF,
                                                           pyramid[level].E);
        }
        else
        {
          krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchSize,
                                                           PatchSSD_Split<NS,NG,float>(pyramid[level].targetStyle,
                                                                                       pyramid[level].sourceStyle,
                                                                                       pyramid[level].targetGuide,
                                                                                       pyramid[level].sourceGuide,
                                                                                       styleWeightsVec,
                                                                                       guideWeightsVec),
                                                           pyramid[level].NNF,
                                                           pyramid[level].E);
        }
        checkCudaError( cudaDeviceSynchronize() );
      }

      {
        const int numThreadsPerBlock = 24;
        const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
        const dim3 numBlocks = dim3((pyramid[level].targetWidth+threadsPerBlock.x)/threadsPerBlock.x,
                                    (pyramid[level].targetHeight+threadsPerBlock.y)/threadsPerBlock.y);

        if      (voteMode==EBSYNTH_VOTEMODE_PLAIN)
        {
          krnlVotePlain<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                       pyramid[level].sourceStyle,
                                                       pyramid[level].NNF,
                                                       patchSize);
        }
        else if (voteMode==EBSYNTH_VOTEMODE_WEIGHTED)
        {
          krnlVoteWeighted<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                          pyramid[level].sourceStyle,
                                                          pyramid[level].NNF,
                                                          pyramid[level].E,
                                                          patchSize);
        }

        std::swap(pyramid[level].targetStyle2,pyramid[level].targetStyle);
        checkCudaError( cudaDeviceSynchronize() );

        if (voteIter<numSearchVoteItersPerLevel[level]-1)
        {
          krnlEvalMask<<<numBlocks,threadsPerBlock>>>(pyramid[level].mask,
                                                      pyramid[level].targetStyle,
                                                      pyramid[level].targetStyle2,
                                                      stopThresholdPerLevel[level]);
          checkCudaError( cudaDeviceSynchronize() );

          krnlDilateMask<<<numBlocks,threadsPerBlock>>>(pyramid[level].mask2,
                                                        pyramid[level].mask,
                                                        patchSize);
          std::swap(pyramid[level].mask2,pyramid[level].mask);
          checkCudaError( cudaDeviceSynchronize() );
        }
      }
    }
  }

  checkCudaError( cudaDeviceSynchronize() );

  copy(&outputData,pyramid[pyramid.size()-1].targetStyle);

  checkCudaError( cudaFree(rngStates) );

  for(int level=0;level<pyramid.size();level++)
  {
    pyramid[level].sourceStyle.destroy();
    pyramid[level].sourceGuide.destroy();
    pyramid[level].targetStyle.destroy();
    pyramid[level].targetStyle2.destroy();
    pyramid[level].mask.destroy();
    pyramid[level].mask2.destroy();
    pyramid[level].targetGuide.destroy();
    pyramid[level].NNF.destroy();
    pyramid[level].NNF2.destroy();
    pyramid[level].E.destroy();
    pyramid[level].Omega.destroy();
    if (targetModulationData) { pyramid[level].targetModulation.destroy(); }
  }
}

EBSYNTH_API void ebsynthRun(int    ebsynthBackend,
                            int    numStyleChannels,
                            int    numGuideChannels,
                            int    sourceWidth,
                            int    sourceHeight,
                            void*  sourceStyleData,
                            void*  sourceGuideData,
                            int    targetWidth,
                            int    targetHeight,
                            void*  targetGuideData,
                            void*  targetModulationData,
                            float* styleWeights,
                            float* guideWeights,
                            float  uniformityWeight,
                            int    patchSize,
                            int    voteMode,
                            int    numPyramidLevels,
                            int*   numSearchVoteItersPerLevel,
                            int*   numPatchMatchItersPerLevel,
                            int*   stopThresholdPerLevel,
                            void*  outputData
                            )
{
  void(*const dispatchEbsynth[EBSYNTH_MAX_GUIDE_CHANNELS])(int, int, int, int, int, void*, void*, int, int, void*, void*, float*, float*, float, int, int, int, int*, int*, int*, void*) =
  {
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,1>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,2>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,3>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,4>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,5>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,6>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,7>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,8>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,9>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,10>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,11>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,12>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,13>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,14>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,15>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,16>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,17>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,18>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,19>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,20>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,21>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,22>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,23>,
	  runEbsynth<EBSYNTH_NUM_STYLE_CHANNELS,24>,
  };

  if (numStyleChannels != EBSYNTH_NUM_STYLE_CHANNELS) { printf("ebsynth.dll error: this version only supports exactly %d style channels!\n", EBSYNTH_NUM_STYLE_CHANNELS); return; }
  if (numGuideChannels<1) { printf("ebsynth.dll error: expecting at least one guide channel!\n"); return; }
  if (numGuideChannels>EBSYNTH_MAX_GUIDE_CHANNELS) { printf("ebsynth.dll error: too many guide channels!\n"); return; }

  if (numGuideChannels>=1 && numGuideChannels<=EBSYNTH_MAX_GUIDE_CHANNELS)
  {
    dispatchEbsynth[numGuideChannels-1](ebsynthBackend,
                                        numStyleChannels,
                                        numGuideChannels,
                                        sourceWidth,
                                        sourceHeight,
                                        sourceStyleData,
                                        sourceGuideData,
                                        targetWidth,
                                        targetHeight,
                                        targetGuideData,
                                        targetModulationData,
                                        styleWeights,
                                        guideWeights,
                                        uniformityWeight,
                                        patchSize,
                                        voteMode,
                                        numPyramidLevels,
                                        numSearchVoteItersPerLevel,
                                        numPatchMatchItersPerLevel,
                                        stopThresholdPerLevel,
                                        outputData);
  }
}

EBSYNTH_API
int ebsynthBackendAvailable(int ebsynthBackend)
{
  if (ebsynthBackend==EBSYNTH_BACKEND_CUDA)
  {
    int deviceCount = -1;
    if (cudaGetDeviceCount(&deviceCount)!=cudaSuccess) { return 0; }

    for (int device=0;device<deviceCount;device++)
    {
      cudaDeviceProp properties;
      if (cudaGetDeviceProperties(&properties,device)==cudaSuccess)
      {
        if (properties.major!=9999 && properties.major>=3)
        {
          return 1;
        }
      }
    }
  }

  return 0;
}
