#include "LPSO_kernels.cuh"

#include <algorithm>


extern "C" __device__ __device_builtin__ void __syncthreads();
extern "C" __device__ __device_builtin__ float fminf(float x, float y);
extern "C" __device__ __device_builtin__ float fmaxf(float x, float y);
extern "C" __device__ __device_builtin__ unsigned int __uAtomicInc(unsigned int *address, unsigned int val);
extern "C" __device__ __device_builtin__ void __threadfence_system(void);


template<class T>
__global__ void k_LPSOInit(
	int numParticles, 
	int numDimensions, 
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	curandState *_s)
{
	__shared__ float bestFitness[1024];
	__shared__ T Eval;

	int idx = threadIdx.x;
	int ptr_g0 = blockDim.x * blockIdx.x;
	int gidx = ptr_g0 + idx;
	if (gidx >= numParticles)
		bestFitness[idx] = FLT_MAX;
	__syncthreads();

	int ptr_g = gidx * numDimensions; // posição na memoria
	if (gidx < numParticles)
	{
		curand_init(threadIdx.x, 0, 0, &_s[idx]);

		// Calculate randon pos & vel
		for (int d = 0; d < numDimensions; ++d)
		{
			float min = _c_LPSOMinPosition[d];
			float max = _c_LPSOMaxPosition[d];
			_positions[ptr_g + d] = curand_uniform(&_s[idx])*(max - min) + min;
			_velocities[ptr_g + d] = curand_uniform(&_s[idx])*(max - min) + min;
		}

		// Initizalizes local bests
		bestFitness[idx] = Eval(&_positions[ptr_g]);
	}
	__syncthreads();

	if (gidx < numParticles)
	{
		for (int d = 0; d < numDimensions; ++d)
			_bestPositions[ptr_g + d] = _positions[ptr_g + d];
		_bestFitness[gidx] = bestFitness[idx];
	}
}

template<class T>
__global__ void k_LPSOIterateMultiBlock(
	int numParticles, 
	int numDimensions, 
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	curandState *_s)
{
	__shared__ T Eval;

	int p = threadIdx.x;
	int block = blockIdx.x;
	int ptr_g0 = blockDim.x * block;
	int gp = ptr_g0 + p;
	int lastP = fminf(numParticles - ptr_g0, 1024);

	__syncthreads();

	int ptr_g = gp * numDimensions;
	if (gp < numParticles)
	{
		int prev = (p + numParticles - 1) % numParticles;
		int next = (p + 1) % numParticles;
		int best = gp;
		if (_bestFitness[prev] < _bestFitness[best])
			best = prev;
		if (_bestFitness[next] < _bestFitness[best])
			best = prev;

		for (int d = 0; d < numDimensions; ++d)
		{
			float r1 = curand_uniform(&_s[p]);
			float r2 = curand_uniform(&_s[p]);

			int ptr = ptr_g + d;
			float position = _positions[ptr];
			float newVelocity = (W * _velocities[ptr]) +
				(C1 * r1 * (_bestPositions[ptr] - position)) +
				(C2 * r2 * (_bestPositions[best * numDimensions + d] - position));

			newVelocity = fmaxf(_c_LPSOMinPosition[d], fminf(_c_LPSOMaxPosition[d], newVelocity));
			_velocities[ptr] = newVelocity;

			float newPosition = position + newVelocity;
			newPosition = fmaxf(_c_LPSOMinPosition[d], fminf(_c_LPSOMaxPosition[d], newPosition));
			_positions[ptr] = newPosition;
		}
		float newFitness = Eval(&_positions[ptr_g]);
		if (newFitness < _bestFitness[gp])
		{
			_bestFitness[gp] = newFitness;
			for (int d = 0; d < numDimensions; ++d)
			{
				int ptr = ptr_g + d;
				_bestPositions[ptr] = _positions[ptr];
			}
		}
	}
	__syncthreads();
}


__global__ void k_LPSOInterMinimum(
	int numParticles, 
	int numDimensions, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s)
{
	__shared__ int ptrs[1024];
	__shared__ float bestFitness[1024];

	int p = threadIdx.x;
	int block = blockIdx.x;
	int ptr_g0 = blockDim.x * block;
	int gp = ptr_g0 + p;
	if (gp < numParticles)
		bestFitness[p] = _bestFitness[gp];
	else
		bestFitness[p] = FLT_MAX;
	__syncthreads();

	ptrs[p] = p;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s /= 2)
	{
		if (p < s)
		{
			if (bestFitness[ptrs[p]] > bestFitness[ptrs[p + s]])
			{
				int tmp = ptrs[p + s];
				ptrs[p + s] = ptrs[p];
				ptrs[p] = tmp;
			}
		}
		__syncthreads();
	}
	
	if (p < numDimensions)
		_bestGlobalPosition[block * numDimensions + p] = _bestPositions[(ptr_g0 + ptrs[0]) * numDimensions + p];
	if (p == 0)
		_bestGlobalFitness[block] = bestFitness[ptrs[0]];
}


__global__ void k_LPSOMinimum(int _numBlocks, int numDimensions, float *_position, float *_fitness)
{
	__shared__ float fitness[1024];
	__shared__ int ptrs[1024];

	int idx = threadIdx.x;
	ptrs[idx] = idx;
	if (idx >= _numBlocks)
		fitness[idx] = FLT_MAX;
	__syncthreads();

	if (idx < _numBlocks)
		fitness[idx] = _fitness[idx];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s /= 2)
	{
		if (idx < s)
		{
			if (fitness[ptrs[idx]] > fitness[ptrs[idx + s]])
			{
				int tmp = ptrs[idx + s];
				ptrs[idx + s] = ptrs[idx];
				ptrs[idx] = tmp;
			}
		}
		__syncthreads();
	}
	if (idx < numDimensions)
		_position[idx] = _position[ptrs[0] * numDimensions + idx];
	if (idx == 0)
		_fitness[0] = _fitness[ptrs[0]];
}
