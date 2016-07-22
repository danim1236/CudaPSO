#include "PSOCpp.h"

#include <iostream>
#include <time.h>

int main(int argc, char**argv)
{
	float min[2] = { -10.0, -10.0 };
	float max[2] = { 10.0, 10.0 };

	PSOCpp pso(512, sizeof(min) / sizeof(float), min, max);
	pso.Init();
	pso.PrintHeader();

	for(int i=0;i<20;++i) {
		clock_t start = clock();

		pso.Iterate(50);

		clock_t stop = clock();

		float elapsedTime = 1000.0 * (stop - start) / CLOCKS_PER_SEC;
		pso.PrintStatus(elapsedTime);
	}
	while(1);
}
