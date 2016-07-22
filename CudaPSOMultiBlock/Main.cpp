#include "PSOCuda.cuh"

#include <iostream>

int main(int argc, char**argv)
{
	float min[2] = { -10.0, -10.0 };
	float max[2] = { 10.0, 10.0 };

	int numDimensions = sizeof(min) / sizeof(float);
	PSOCuda pso(51200, numDimensions, min, max);
	pso.Init();
	pso.PrintHeader();

	for(int i=0;i<1;++i) {
		cudaEvent_t start, stop;
		float elapsedTime;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		pso.Iterate(200);

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start,stop);

		float *best = pso.GetBestPosition();
		float stdDev = pso.GetStdDev();
		pso.PrintStatus(elapsedTime);
	}
	while (true);
}
