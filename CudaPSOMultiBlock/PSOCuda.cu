#include "PSOCuda.cuh"

#include <stdexcept>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/functional.h>
#include <thrust/extrema.h>


extern "C" __device__ __device_builtin__ void __syncthreads();
extern "C" __device__ __device_builtin__ float fminf(float x, float y);
extern "C" __device__ __device_builtin__ float fmaxf(float x, float y);
extern "C" __device__ __device_builtin__ unsigned int __uAtomicInc(unsigned int *address, unsigned int val);
extern "C" __device__ __device_builtin__ void __threadfence_system(void);

#define MAX_DIMENSIONS 20

__constant__ float _c_minPosition[MAX_DIMENSIONS];
__constant__ float _c_maxPosition[MAX_DIMENSIONS];

__forceinline__ __device__ float EvalBanana(float *position)
{
	float x = position[0];
	float y = position[1];
	float a = y - x * x;
	float b = 1 - x;
	return 100 * a*a + b*b;	
}


__global__ void k_InitPSO(
	int numParticles, 
	int numDimensions, 
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s)
{
	__shared__ float bestFitness[1024];
	__shared__ int ptrs[1024];

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
			float min = _c_minPosition[d];
			float max = _c_maxPosition[d];
			_positions[ptr_g + d] = curand_uniform(&_s[idx])*(max - min) + min;
			_velocities[ptr_g + d] = curand_uniform(&_s[idx])*(max - min) + min;
		}

		// Initizalizes local bests
		bestFitness[idx] = EvalBanana(&_positions[ptr_g]);
	}
	__syncthreads();

	// Descobre a melhor
	ptrs[idx] = idx;
	for (int s = 1024 / 2; s > 0; s /= 2)
	{
		if (idx < s)
		{
			if (bestFitness[ptrs[idx]] > bestFitness[ptrs[idx + s]])
			{
				int tmp = ptrs[idx + s];
				ptrs[idx + s] = ptrs[idx];
				ptrs[idx] = tmp;
			}
		}
		__syncthreads();
	}

	if (gidx < numParticles)
	{
		for (int d = 0; d < numDimensions; ++d)
			_bestPositions[ptr_g + d] = _positions[ptr_g + d];
		_bestFitness[gidx] = bestFitness[idx];
		if (idx < numDimensions)
			_bestGlobalPosition[blockIdx.x * numDimensions + idx] = _positions[(ptr_g0 + ptrs[0]) * numDimensions + idx];
		if (idx == 0)
			_bestGlobalFitness[blockIdx.x] = bestFitness[ptrs[0]];
	}
}

__global__ void k_IterateMultiBlock(
	int numParticles, 
	int numDimensions, 
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s)
{
	__shared__ int ptrs[1024];
	__shared__ float bestFitness[1024];
	float bestGlobalFitness;

	int p = threadIdx.x;
	int block = blockIdx.x;
	int ptr_g0 = blockDim.x * block;
	int gp = ptr_g0 + p;
	if (gp < numParticles)
	{
		bestFitness[p] = _bestFitness[gp];
	}

	if (p == 0)
		bestGlobalFitness = _bestGlobalFitness[0];
	else if (gp >= numParticles)
		bestFitness[p] = FLT_MAX;
	__syncthreads();

	int ptr_g = gp * numDimensions;
	if (gp < numParticles)
	{
		for (int d = 0; d < numDimensions; ++d)
		{
			float r1 = curand_uniform(&_s[p]);
			float r2 = curand_uniform(&_s[p]);

			int ptr = ptr_g + d;
			float position = _positions[ptr];
			float newVelocity = (W * _velocities[ptr]) +
				(C1 * r1 * (_bestPositions[ptr] - position)) +
				(C2 * r2 * (_bestGlobalPosition[block * numDimensions + d] - position));

			newVelocity = fmaxf(_c_minPosition[d], fminf(_c_maxPosition[d], newVelocity));
			_velocities[ptr] = newVelocity;

			float newPosition = position + newVelocity;
			newPosition = fmaxf(_c_minPosition[d], fminf(_c_maxPosition[d], newPosition));
			_positions[ptr] = newPosition;
		}
		float newFitness = EvalBanana(&_positions[ptr_g]);
		if (newFitness < bestFitness[p])
		{
			bestFitness[p] = newFitness;
			for (int d = 0; d < numDimensions; ++d)
			{
				int ptr = ptr_g + d;
				_bestPositions[ptr] = _positions[ptr];
			}
		}
	}
		__syncthreads();

	// Descobre a melhor
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
	if (p == 0)
	{
		if (bestFitness[ptrs[0]] < bestGlobalFitness)
		{
			bestGlobalFitness = bestFitness[ptrs[0]];
			for (int d = 0; d < numDimensions; ++d)
				_bestGlobalPosition[block * numDimensions + d] = _positions[(ptr_g0 + ptrs[0]) * numDimensions + d];
		}
	}
	__syncthreads();
	
	if (gp < numParticles)
		_bestFitness[gp] = bestFitness[p];
	if (p == 0)
		_bestGlobalFitness[block] = bestGlobalFitness;
}

__global__ void k_IterateSingleBlock(
	int numParticles, 
	int numDimensions, 
	int numIterations,
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s)
{
	__shared__ int ptrs[1024];
	__shared__ float bestFitness[1024];
	float bestGlobalFitness;

	int p = threadIdx.x;
	int block = blockIdx.x;
	if (p < numParticles)
		bestFitness[p] = _bestFitness[p];

	if (p == 0)
		bestGlobalFitness = _bestGlobalFitness[0];
	else if (p >= numParticles)
		bestFitness[p] = FLT_MAX;
	__syncthreads();

	int ptr_g = p * numDimensions;
	for (int it = 0; it < numIterations; ++it)
	{
		if (p < numParticles)
		{
			for (int d = 0; d < numDimensions; ++d)
			{
				float r1 = curand_uniform(&_s[p]);
				float r2 = curand_uniform(&_s[p]);

				int ptr = ptr_g + d;

				float position = _positions[ptr];
				float newVelocity = (W * _velocities[ptr]) +
					(C1 * r1 * (_bestPositions[ptr] - position)) +
					(C2 * r2 * (_bestGlobalPosition[block * numDimensions + d] - position));

				newVelocity = fmaxf(_c_minPosition[d], fminf(_c_maxPosition[d], newVelocity));
				_velocities[ptr] = newVelocity;

				float newPosition = position + newVelocity;
				newPosition = fmaxf(_c_minPosition[d], fminf(_c_maxPosition[d], newPosition));
				_positions[ptr] = newPosition;
			}
			float newFitness = EvalBanana(&_positions[p * numDimensions]);
			if (newFitness < bestFitness[p])
			{
				bestFitness[p] = newFitness;
				for (int d = 0; d < numDimensions; ++d)
				{
					_bestPositions[ptr_g + d] = _positions[ptr_g + d];
				}
			}
		}
			__syncthreads();

		// Descobre a melhor
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
	}

	if (p == 0)
	{
		if (bestFitness[ptrs[0]] < bestGlobalFitness)
		{
			bestGlobalFitness = bestFitness[ptrs[0]];
			for (int d = 0; d < numDimensions; ++d)
			{
				_bestGlobalPosition[block * numDimensions + d] = _positions[ptrs[0] * numDimensions + d];
			}
		}
	}
	__syncthreads();
	
	if (p < numParticles)
		_bestFitness[p] = bestFitness[p];
	if (p == 0)
		_bestGlobalFitness[block] = bestGlobalFitness;
}


__global__ void k_minimum(int _numBlocks, int numDimensions, float *_position, float *_fitness)
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

PSOCuda::PSOCuda(int numParticles, int numDimensions, float *minPositions, float *maxPositions)
:
PSOBase(numParticles, numDimensions, minPositions, maxPositions),
_d_positions(_positions.size()),
_d_velocities(_velocities.size()),
_d_minPositions(_minPositions),
_d_maxPositions(_maxPositions),
_d_bestPositions(_bestPositions.size()),
_d_bestFitness(_bestFitness.size()),
_d_state(numParticles)
{
	if (_numDimensions > MAX_DIMENSIONS)
		throw new exception("_numDimensions > MAX_DIMENSIONS");
	CalculateGeometry();
	_d_bestGlobalPosition.resize(_numDimensions * _numBlocks);
	_d_bestGlobalFitness.resize(_numBlocks);
	_bestGlobalPosition.resize(_numDimensions * _numBlocks);
	_bestGlobalFitness.resize(_numBlocks);
	cudaMemcpyToSymbol(_c_minPosition, _minPositions.data(), _minPositions.size() * sizeof(float));
	cudaMemcpyToSymbol(_c_maxPosition, _maxPositions.data(), _maxPositions.size() * sizeof(float));
}

void PSOCuda::Init()
{
	int threadNumber = pow(2, ceil(log(_numThreads)/log(2)));
	int blockNumber = pow(2, ceil(log(_numBlocks)/log(2)));
	k_InitPSO<<<_numBlocks, threadNumber>>>(_numParticles, _numDimensions,
		raw_pointer_cast(_d_positions.data()), 
		raw_pointer_cast(_d_velocities.data()), 
		raw_pointer_cast(_d_bestPositions.data()),
		raw_pointer_cast(_d_bestFitness.data()),
		raw_pointer_cast(_d_bestGlobalPosition.data()),
		raw_pointer_cast(_d_bestGlobalFitness.data()),
		raw_pointer_cast(_d_state.data()));
	cudaDeviceSynchronize();
	k_minimum<<<1, blockNumber>>>(_numBlocks, _numDimensions,
		raw_pointer_cast(_d_bestGlobalPosition.data()),
		raw_pointer_cast(_d_bestGlobalFitness.data()));
	UpdateHost();
}

void PSOCuda::Iterate(int n)
{
	int threadNumber = pow(2, ceil(log(_numThreads)/log(2)));
	int blockNumber = pow(2, ceil(log(_numBlocks)/log(2)));
	if (blockNumber == 1)
	{
		k_IterateSingleBlock<<<_numBlocks, threadNumber>>>(_numParticles, _numDimensions, n,
			raw_pointer_cast(_d_positions.data()), 
			raw_pointer_cast(_d_velocities.data()), 
			raw_pointer_cast(_d_bestPositions.data()),
			raw_pointer_cast(_d_bestFitness.data()),
			raw_pointer_cast(_d_bestGlobalPosition.data()),
			raw_pointer_cast(_d_bestGlobalFitness.data()),
			raw_pointer_cast(_d_state.data()));
	}
	else
	{
		for (int i = 0; i < n; ++i)
		{
			k_IterateMultiBlock<<<_numBlocks, threadNumber>>>(_numParticles, _numDimensions,
				raw_pointer_cast(_d_positions.data()), 
				raw_pointer_cast(_d_velocities.data()), 
				raw_pointer_cast(_d_bestPositions.data()),
				raw_pointer_cast(_d_bestFitness.data()),
				raw_pointer_cast(_d_bestGlobalPosition.data()),
				raw_pointer_cast(_d_bestGlobalFitness.data()),
				raw_pointer_cast(_d_state.data()));
			k_minimum<<<1, blockNumber>>>(_numBlocks, _numDimensions,
				raw_pointer_cast(_d_bestGlobalPosition.data()),
				raw_pointer_cast(_d_bestGlobalFitness.data()));
		}
	}
	_iteration += n;
	UpdateHost();
}

void PSOCuda::UpdateHost()
{
	_positions = _d_positions;
	_velocities = _d_velocities;
	_minPositions = _d_minPositions;
	_maxPositions = _d_maxPositions;
	_bestPositions = _d_bestPositions;
	_bestFitness = _d_bestFitness;
	_bestGlobalPosition = _d_bestGlobalPosition;
	_bestGlobalFitness = _d_bestGlobalFitness;
}

void PSOCuda::CalculateGeometry()
{
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices < 1)
		throw std::exception("Nenhum dispositivo cuda");

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	int maxThreads = devProp.maxThreadsPerBlock;

	_numThreads = (_numParticles + 31 ) / 32 * 32;
	_numThreads = std::min(((_numThreads + 31)/32)*32, maxThreads);
	_numBlocks = (_numParticles + _numThreads - 1) / _numThreads;
}