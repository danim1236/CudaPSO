#ifndef __FUNTORS_CUH__DEFINED__
#define __FUNTORS_CUH__DEFINED__


#include "cuda_runtime.h"

#include <string>

using namespace std;

#define PI ((float)3.1415926535897932)


class EvalBanana
{
public:
	__device__ float operator()(float *position)
	{
		float x = position[0];
		float y = position[1];
		float a = y - x * x;
		float b = 1 - x;
		return 100 * a*a + b*b;	
	}

	static float *GetMinPosition()
	{ 
		return _min;
	}
	
	static float *GetMaxPosition()
	{ 
		return _max;
	}

	static int GetNumDimensions() { return sizeof(_min) / sizeof(float); }

	static string GetName() { return "Banana"; }

private:
	static float _min[2];
	static float _max[2];
};


class EvalF1
{
public:
	__device__ float operator()(float *position)
	{
		float acc = 0;
		for (int i = 0; i < sizeof(_min) / sizeof(float); i++)
		{
			float x = position[i];
			acc += x * x;
		}
		return acc;
	}

	static float *GetMinPosition()
	{
		for (int i = 0; i < sizeof(_min) / sizeof(float); i++)
			_min[i] = -100;
		return _min;
	}

	static float *GetMaxPosition()
	{
		for (int i = 0; i < sizeof(_max) / sizeof(float); i++)
			_max[i] = 100;
		return _max;
	}
	static int GetNumDimensions() { return sizeof(_min) / sizeof(float); }

	static string GetName() { return "F1"; }

private:
	static float _min[50];
	static float _max[50];
};


class EvalF2
{
public:
	__device__ float operator()(float *position)
	{
		float acc = 0;
		for (int i = 0; i < sizeof(_min) / sizeof(float); i++)
		{
			float x = position[i];
			acc += x * x - 10 * cosf(2 * PI * x) + 10;
		}
		return acc;
	}

	static float *GetMinPosition()
	{
		for (int i = 0; i < sizeof(_min) / sizeof(float); i++)
			_min[i] = -10;
		return _min;
	}

	static float *GetMaxPosition()
	{
		for (int i = 0; i < sizeof(_max) / sizeof(float); i++)
			_max[i] = 10;
		return _max;
	}
	static int GetNumDimensions() { return sizeof(_min) / sizeof(float); }

	static string GetName() { return "F2"; }

private:
	static float _min[50];
	static float _max[50];
};


class EvalF3
{
public:
	__device__ float operator()(float *position)
	{
		float sum = 0;
		float prod = 1;
		for (int i = 0; i < sizeof(_min) / sizeof(float); i++)
		{
			float x = position[i];
			sum += x*x;
			prod *= cosf(x / sqrtf(i + 1));
		}
		return sum / 4000 - prod + 1;
	}

	static float *GetMinPosition()
	{
		for (int i = 0; i < sizeof(_min) / sizeof(float); i++)
			_min[i] = -600;
		return _min;
	}

	static float *GetMaxPosition()
	{
		for (int i = 0; i < sizeof(_max) / sizeof(float); i++)
			_max[i] = 600;
		return _max;
	}
	static int GetNumDimensions() { return sizeof(_min) / sizeof(float); }

	static string GetName() { return "F3"; }

private:
	static float _min[50];
	static float _max[50];
};


class EvalF4
{
public:
	__device__ float operator()(float *position)
	{
		float acc = 0;
		for (int i = 0; i < sizeof(_min) / sizeof(float) - 1; i++)
		{
			float x = position[i];
			float y = position[i + 1];
			acc += 100 * (y - x*x)*(y - x*x) + (x - 1)*(x - 1);
		}
		return acc;
	}

	static float *GetMinPosition()
	{
		for (int i = 0; i < sizeof(_min) / sizeof(float); i++)
			_min[i] = -10;
		return _min;
	}

	static float *GetMaxPosition()
	{
		for (int i = 0; i < sizeof(_max) / sizeof(float); i++)
			_max[i] = 10;
		return _max;
	}
	static int GetNumDimensions() { return sizeof(_min) / sizeof(float); }

	static string GetName() { return "F4"; }

private:
	static float _min[50];
	static float _max[50];
};


#endif