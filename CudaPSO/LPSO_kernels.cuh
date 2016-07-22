#ifndef __LPSO_KERNELS_CUH__DEFINED__
#define __LPSO_KERNELS_CUH__DEFINED__


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>


#define MAX_DIMENSIONS 1024

#define W (float)0.729
#define C1 (float)1.49445
#define C2 (float)1.49445


__constant__ float _c_LPSOMinPosition[MAX_DIMENSIONS];
__constant__ float _c_LPSOMaxPosition[MAX_DIMENSIONS];


template<class T>
__global__ void k_LPSOInit(
	int numParticles, 
	int numDimensions, 
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	curandState *_s);

template<class T>
__global__ void k_LPSOIterateMultiBlock(
	int numParticles, 
	int numDimensions, 
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	curandState *_s);

__global__ void k_LPSOInterMinimum(
	int numParticles, 
	int numDimensions, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s);

__global__ void k_LPSOMinimum(
	int _numBlocks,
	int numDimensions,
	float *_position,
	float *_fitness);


#endif
