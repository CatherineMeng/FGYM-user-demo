
#include "hls_stream.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{
using namespace std;

#define TILES 2 //SIZE/BLOCK_SIZE
#define P 2
#define T 2
//typedef int DTYPE;
const int SIZE = 8;
const int BLOCK_SIZE = 4;
// PE array sizes


typedef struct {
	float a[BLOCK_SIZE];
} blockvec;

typedef struct {
	float out[BLOCK_SIZE][BLOCK_SIZE];
} blockmat;

void loadDDR(blockvec A[], blockvec B[], hls::stream<blockvec> &Arows, hls::stream<blockvec> &Bcols, int it);
void blockmatmul(hls::stream<blockvec> &Arows, hls::stream<blockvec> &Bcols, hls::stream<blockmat> &Cblocks, int it);
void storeDDR(blockmat C[],  hls::stream<blockmat> &Cblocks,  int it);
void top(blockvec *A, blockvec *B, blockmat *C);
}