#pragma once

#include "PSOBase.h"

#include "TAPSO_kernels.cuh"

#include <thrust\device_vector.h>
#include <memory>
#include <curand.h>
#include <curand_kernel.h>

#include <string>

using namespace std;
using namespace thrust;

template<typename T>
class TAPSO : public PSOBase
{
public:
	TAPSO(int numParticles, int numIterations);
	
	void Init();
	void Iterate();
	
	string GetName() { return T::GetName(); }

private:
	int _numDevices;
	int _maxThreads;
	int _maxBlocks;

	int _numThreads;
	int _numBlocks;

	device_vector<float> _d_positions;
	device_vector<float> _d_velocities;

	device_vector<float> _d_minPositions;
	device_vector<float> _d_maxPositions;

	device_vector<float> _d_bestPositions;
	device_vector<float> _d_bestFitness;

	device_vector<float> _d_bestGlobalPosition;
	device_vector<float> _d_bestGlobalFitness;

	device_vector<curandState> _d_state;

	void CalculateGeometry();
	void UpdateHost();
};
