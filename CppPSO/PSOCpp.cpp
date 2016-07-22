#include "PSOCpp.h"


#include <algorithm>

#define Rand() (rand() / (float)(RAND_MAX + 1))
#define NUM_THREADS 4

void PSOCpp::Init()
{
	#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
	for (int p = 0; p < _numParticles; ++p)
	{
		float *positions = &_positions[p * _numDimensions];
		float *bestPositions = &_positions[p * _numDimensions];
		float *velocities = &_positions[p * _numDimensions];
		for (int j = 0; j < _numDimensions; ++j)
		{
			bestPositions[j] = positions[j] = Rand()*(_maxPositions[j] - _minPositions[j]) + _minPositions[j];
			velocities[j] = Rand()*(_maxPositions[j] - _minPositions[j]) + _minPositions[j];
		}
		float fitness = Eval(positions);
		_bestFitness[p] = fitness;
		if (fitness < _bestGlobalFitness[0])
		{
			_bestGlobalFitness[0] = fitness;
			memcpy(&_bestGlobalPosition[0], positions, _numDimensions * sizeof(float));
		}
	}
}

void PSOCpp::Iterate(int numIterations)
{
	while (numIterations-- > 0)
	{
		#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
		for (int p = 0; p < _numParticles; ++p)
		{
			for (int j = 0; j < _numDimensions; ++j)
			{
				float r1 = Rand();
				float r2 = Rand();

				float newVelocity = (W * _velocities[p * _numDimensions + j]) +
					(C1 * r1 * (_bestPositions[p * _numDimensions + j] - _positions[p * _numDimensions + j])) +
					(C2 * r2 * (_bestGlobalPosition[j] - _positions[p * _numDimensions + j]));
					           
				newVelocity = std::max(-_maxPositions[j], std::min(_maxPositions[j], newVelocity));
				_velocities[p * _numDimensions + j] = newVelocity;
					
				float newPosition = _positions[p * _numDimensions + j] + newVelocity;
				newPosition = std::max(_minPositions[j], std::min(_maxPositions[j], newPosition));
				_positions[p * _numDimensions + j] = newPosition;
			}
			float newFitness = Eval(&_positions[p * _numDimensions]);
			if (newFitness < _bestFitness[p])
			{
				_bestFitness[p] = newFitness;
				for (int j = 0; j < _numDimensions; ++j)
				{
					_bestPositions[p * _numDimensions + j] = _positions[p * _numDimensions + j];
				}
			}
		}

		#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
		for (int p = 0; p < _numParticles; ++p)
		{
			if (_bestFitness[p] < _bestGlobalFitness[0])
			{
				_bestGlobalFitness[0] = _bestFitness[p];
				for (int j = 0; j < _numDimensions; ++j)
				{
					_bestGlobalPosition[j] = _positions[p * _numDimensions + j];
				}
			}
		}
		_iteration++;
	}
}


