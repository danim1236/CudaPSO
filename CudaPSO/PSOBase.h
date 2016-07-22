#pragma once

#include <thrust\host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <string>


using namespace std;
using namespace thrust;
using namespace thrust::cuda::experimental;


#define W (float)0.729
#define C1 (float)1.49445
#define C2 (float)1.49445


class PSOBase
{
public:
	PSOBase(int numParticles, int numIterations, int numDimensions, float *minPositions, float *maxPositions);

	virtual void Init() = 0;
	virtual void Iterate() = 0;
	
	float* GetPosition(int particle) { return &_positions[particle * _numDimensions]; }
	float* GetBestPosition() { return &_bestGlobalPosition[0]; }
	float* GetVelocity(int particle) { return &_velocities[particle * _numDimensions]; }
	
	virtual float GetStdDev();
	void PrintHeader();
	void PrintHeaderFull();
	void PrintStatus(float elapsedTime);
	void PrintStatusFull(float elapsedTime);

	virtual string GetName() = 0;

protected:
	int _numParticles;
	int _numIterations;
	int _numDimensions;

	int _iteration;

	host_vector<float, pinned_allocator<float>> _positions;
	host_vector<float, pinned_allocator<float>> _velocities;

	host_vector<float, pinned_allocator<float>> _minPositions;
	host_vector<float, pinned_allocator<float>> _maxPositions;

	host_vector<float, pinned_allocator<float>> _bestPositions;
	host_vector<float, pinned_allocator<float>> _bestFitness;

	host_vector<float, pinned_allocator<float>> _bestGlobalPosition;
	host_vector<float, pinned_allocator<float>> _bestGlobalFitness;
};

