#include "PSOBase.h"

#include <cstdlib> 
#include <ctime>
#include <math.h>


using namespace std;


PSOBase::PSOBase(int numParticles, int numDimensions, float *minPositions, float *maxPositions)
:
_numParticles(numParticles),
_numDimensions(numDimensions),
_iteration(0),
_positions(numParticles * numDimensions, 0.0),
_velocities(numParticles * numDimensions, 0.0),
_minPositions(numDimensions),
_maxPositions(numDimensions),
_bestPositions(numParticles * numDimensions, 0.0),
_bestFitness(numParticles),
_bestGlobalPosition(numDimensions, 0.0),
_bestGlobalFitness(1, FLT_MAX)
{
	srand((unsigned) time(NULL));

	memcpy(&_minPositions[0], minPositions, _minPositions.size() * sizeof(float));
	memcpy(&_maxPositions[0], maxPositions, _maxPositions.size() * sizeof(float));
}

float PSOBase::GetStdDev()
{
	float result = 0.0;

	for (int i = 0; i < _numDimensions; ++i)
	{
		float mean = 0;
		for (unsigned j = 0; j < _positions.size(); j += _numDimensions) 
		{
			mean += _positions[j];
		}
		mean /= _numParticles;
		float var = 0;
		for (unsigned j = 0; j < _positions.size(); j += _numDimensions) 
		{
			float diff = _positions[j] - mean;
			var += diff * diff;
		}
		var /= (_numParticles - 1);
		result += sqrt(var);

	}

	return result / _numDimensions;
}

void PSOBase::PrintHeader()
{
	cout << "N\tValue\t";
	for (int i = 0; i < _numDimensions; ++i)
		cout << 'X' << i << '\t';
	cout << "stdDev\tElapsed time (ms)" << endl;
}

void PSOBase::PrintStatus(float elapsedTime)
{
	float *best = GetBestPosition();
	float stdDev = GetStdDev();

	cout << _iteration << '\t' << _bestGlobalFitness[0] << '\t';
	for (int i = 0; i < _numDimensions; ++i)
		printf("%0.05f\t",best[i]);
	cout << stdDev << '\t' << elapsedTime << endl;
}
