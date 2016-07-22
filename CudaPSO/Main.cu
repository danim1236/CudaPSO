//#define USE_LPSO
//#define USE_TAPSO
//#define USE_TPSO

#include "LPSO.cu"
#include "TPSO.cu"
#include "TAPSO.cu"

#include <iostream>

int main(int argc, char**argv)
{
	PSOBase *psoList[] = 
	{
#ifdef USE_LPSO
		new LPSO<EvalBanana>(400, 2000),
		new LPSO<EvalBanana>(1200, 2000),
		new LPSO<EvalBanana>(2000, 2000),
		new LPSO<EvalBanana>(2800, 2000),
		// F1
		new LPSO<EvalF1>(400, 2000),
		new LPSO<EvalF1>(1200, 2000),
		new LPSO<EvalF1>(2000, 2000),
		new LPSO<EvalF1>(2800, 2000),
		// F2
		new LPSO<EvalF2>(400, 2000),
		new LPSO<EvalF2>(1200, 2000),
		new LPSO<EvalF2>(2000, 2000),
		new LPSO<EvalF2>(2800, 2000),
		// F3
		new LPSO<EvalF3>(400, 2000),
		new LPSO<EvalF3>(1200, 2000),
		new LPSO<EvalF3>(2000, 2000),
		new LPSO<EvalF3>(2800, 2000),
		// F4
		new LPSO<EvalF4>(400, 10000),
		new LPSO<EvalF4>(1200, 10000),
		new LPSO<EvalF4>(2000, 10000),
		new LPSO<EvalF4>(2800, 10000),
#elif defined USE_TAPSO  
		new TAPSO<EvalBanana>(400, 2000),
		new TAPSO<EvalBanana>(1200, 2000),
		new TAPSO<EvalBanana>(2000, 2000),
		new TAPSO<EvalBanana>(2800, 2000),
		// F1
		new TAPSO<EvalF1>(400, 2000),
		new TAPSO<EvalF1>(1200, 2000),
		new TAPSO<EvalF1>(2000, 2000),
		new TAPSO<EvalF1>(2800, 2000),
		// F2
		new TAPSO<EvalF2>(400, 2000),
		new TAPSO<EvalF2>(1200, 2000),
		new TAPSO<EvalF2>(2000, 2000),
		new TAPSO<EvalF2>(2800, 2000),
		// F3
		new TAPSO<EvalF3>(400, 2000),
		new TAPSO<EvalF3>(1200, 2000),
		new TAPSO<EvalF3>(2000, 2000),
		new TAPSO<EvalF3>(2800, 2000),
		// F4
		new TAPSO<EvalF4>(400, 10000),
		new TAPSO<EvalF4>(1200, 10000),
		new TAPSO<EvalF4>(2000, 10000),
		new TAPSO<EvalF4>(2800, 10000),
#elif defined USE_TPSO 
		new TPSO<EvalBanana>(400, 2000),
		new TPSO<EvalBanana>(1200, 2000),
		new TPSO<EvalBanana>(2000, 2000),
		new TPSO<EvalBanana>(2800, 2000),
		// F1
		new TPSO<EvalF1>(400, 2000),
		new TPSO<EvalF1>(1200, 2000),
		new TPSO<EvalF1>(2000, 2000),
		new TPSO<EvalF1>(2800, 2000),
		// F2
		new TPSO<EvalF2>(400, 2000),
		new TPSO<EvalF2>(1200, 2000),
		new TPSO<EvalF2>(2000, 2000),
		new TPSO<EvalF2>(2800, 2000),
		// F3
		new TPSO<EvalF3>(400, 2000),
		new TPSO<EvalF3>(1200, 2000),
		new TPSO<EvalF3>(2000, 2000),
		new TPSO<EvalF3>(2800, 2000),
		// F4
		new TPSO<EvalF4>(400, 10000),
		new TPSO<EvalF4>(1200, 10000),
		new TPSO<EvalF4>(2000, 10000),
		new TPSO<EvalF4>(2800, 10000),
#else
		new TPSO<EvalBanana>(51200, 50),
#endif
	};

	for (int i = 0; i < sizeof(psoList) / sizeof(PSOBase *); ++i)
	{
		PSOBase &pso(*psoList[i]);
		cout << "Funcao objetivo: " << pso.GetName() << endl;
		pso.Init();
		pso.PrintHeader();

		while (true){
			cudaEvent_t start, stop;
			float elapsedTime;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start,0);

			pso.Iterate();

			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsedTime, start,stop);

			pso.PrintStatus(elapsedTime);
			cout << endl;
		}
	}
	while (true);
}
