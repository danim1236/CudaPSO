#pragma once


#include "PSOBase.h"


class PSOCpp : public PSOBase
{
public:
	PSOCpp(int numParticles, int numDimensions, float *minPositions, float *maxPositions)
	:	PSOBase(numParticles, numDimensions, minPositions, maxPositions)
	{
	}

	void Init();
	void Iterate(int n);

	inline float Eval(float *position)
	{
		float x = position[0];
		float y = position[1];
		float a = y - x*x;
		float b = 1 -x;
		return 100 * a*a + b*b;
	}
};

