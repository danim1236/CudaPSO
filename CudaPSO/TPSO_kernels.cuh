#ifndef __TPSO_KERNELS_CUH__DEFINED__
#define __TPSO_KERNELS_CUH__DEFINED__


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>


#define MAX_DIMENSIONS 1024

#define W (float)0.729
#define C1 (float)1.49445
#define C2 (float)1.49445


__constant__ float _c_TPSOMinPosition[MAX_DIMENSIONS];
__constant__ float _c_TPSOMaxPosition[MAX_DIMENSIONS];


template<typename T>
__global__ void k_TPSOInit(
	int numParticles, 
	int numDimensions, 
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s);

template<typename T>
__global__ void k_TPSOIterateMultiBlock(
	int numParticles, 
	int numDimensions, 
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s);

__global__ void k_TPSOMinimum(
	int _numBlocks,
	int numDimensions,
	float *_position,
	float *_fitness);


#endif
