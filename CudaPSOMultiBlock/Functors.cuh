#include "cuda_runtime.h"
#include "device_launch_parameters.h"


class EvalBase
{
public:
	virtual int GetNumDimensions() = 0;
	virtual float *GetMin() = 0;
	virtual float *GetMax() = 0;
};


class EvalBanana : public EvalBase
{
public:
	__forceinline__ __device__ float operator()(float *position)
	{
		float x = position[0];
		float y = position[1];
		float a = y - x * x;
		float b = 1 - x;
		return 100 * a*a + b*b;	
	}

	float *GetMin()
	{
		static float min[2] = { -10, -10 };
		return min;
	}

	float *GetMax()
	{
		static float max[2] = { -10, -10 };
		return max;
	}

	int GetNumDimensions() { return 2; }
};


#define EvalClass EvalBanana

