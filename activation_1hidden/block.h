
#include "hls_stream.h"
#include "hls_math.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{
using namespace std;


//model metadata
//batchsize:BSIZE last-layer dimension:LL next-layer dimension:LN
//mmult:A(BSIZE,LL)*B(LL,LN)
const int BSIZE = 16;
const int L1 = 4;
const int L2 = 64;
const int L3 = 2;


//hardware parameters
//PE array dimensions
#define P 4
//#define T 32
#define T 16

#define P2 4
#define T2 2
//const int BLOCK_SIZE = 16;
// PE array sizes


typedef struct {
	float a[BSIZE];
} blockvec;

typedef struct {
	float a[L2];
} w1blockvec;
typedef struct {
	float a[L3];
} w3blockvec;
//typedef struct {
//	float a[L4];
//} w3blockvec;
//typedef struct {
//	float out[BLOCK_SIZE][BLOCK_SIZE];
//} blockmat;

void loadIn(blockvec In[],  hls::stream<blockvec> &Inrows,int LL);
//void loadW(w1blockvec W[], hls::stream<blockvec> &Wcols, int LL);
void blockmatmul(hls::stream<blockvec> &Inrows, w1blockvec Wcols[], hls::stream<blockvec> &Crows, int LL,int LN);
void blockmatmul3(hls::stream<blockvec> &Inrows, w3blockvec Wcols[], hls::stream<blockvec> &Crows, int LL,int LN);
void activation(hls::stream<blockvec> &Inrows, float bias[], hls::stream<blockvec> &Outrows,const int L);
void storeDDR(blockvec C[],  hls::stream<blockvec> &Crows,  int LN);
void top(blockvec *A, blockvec *C);
}
