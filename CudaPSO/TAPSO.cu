#include "TAPSO.cuh"

#include "TAPSO_kernels.cu"
#include "Functors.cuh"


#define MUL_INI 2
#define MUL_FIM 0.5


template<typename T>
TAPSO<T>::TAPSO(int numParticles, int numIterations)
:
PSOBase(numParticles, numIterations, T::GetNumDimensions(), T::GetMinPosition(), T::GetMaxPosition()),
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
	cudaMemcpyToSymbol(_c_TAPSOMinPosition, _minPositions.data(), _minPositions.size() * sizeof(float));
	cudaMemcpyToSymbol(_c_TAPSOMaxPosition, _maxPositions.data(), _maxPositions.size() * sizeof(float));
}

template<typename T>
void TAPSO<T>::Init()
{
	int threadNumber = pow(2, ceil(log(_numThreads)/log(2)));
	int blockNumber = pow(2, ceil(log(_numBlocks)/log(2)));
	k_TAPSOInit<T><<<_numBlocks, threadNumber>>>(_numParticles, _numDimensions,
		raw_pointer_cast(_d_positions.data()), 
		raw_pointer_cast(_d_velocities.data()), 
		raw_pointer_cast(_d_bestPositions.data()),
		raw_pointer_cast(_d_bestFitness.data()),
		raw_pointer_cast(_d_bestGlobalPosition.data()),
		raw_pointer_cast(_d_bestGlobalFitness.data()),
		raw_pointer_cast(_d_state.data()));
	cudaDeviceSynchronize();
	k_TAPSOMinimum<<<1, blockNumber>>>(_numBlocks, _numDimensions,
		raw_pointer_cast(_d_bestGlobalPosition.data()),
		raw_pointer_cast(_d_bestGlobalFitness.data()));
	UpdateHost();
}

template<typename T>
void TAPSO<T>::Iterate()
{
	float B = MUL_INI;
	float A = (MUL_FIM - MUL_INI)/(float)(_numIterations-1);

	int threadNumber = pow(2, ceil(log(_numThreads)/log(2)));
	int blockNumber = pow(2, ceil(log(_numBlocks)/log(2)));
	for (int i = 0; i < _numIterations; ++i)
	{
		float k = A*i+B;
		k_TAPSOIterateMultiBlock<T><<<_numBlocks, threadNumber>>>(_numParticles, _numDimensions, k,
			raw_pointer_cast(_d_positions.data()), 
			raw_pointer_cast(_d_velocities.data()), 
			raw_pointer_cast(_d_bestPositions.data()),
			raw_pointer_cast(_d_bestFitness.data()),
			raw_pointer_cast(_d_bestGlobalPosition.data()),
			raw_pointer_cast(_d_bestGlobalFitness.data()),
			raw_pointer_cast(_d_state.data()));
		if (blockNumber > 1)
		{
			k_TAPSOMinimum<<<1, blockNumber>>>(_numBlocks, _numDimensions,
				raw_pointer_cast(_d_bestGlobalPosition.data()),
				raw_pointer_cast(_d_bestGlobalFitness.data()));
		}
	}
	_iteration += _numIterations;
	UpdateHost();
}

template<typename T>
void TAPSO<T>::UpdateHost()
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

template<typename T>
void TAPSO<T>::CalculateGeometry()
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
