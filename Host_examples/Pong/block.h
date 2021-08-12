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
const int BSIZE = 1;
const int L1_total = 12000;
const int L1 = 1200;
const int L2 = 512;
const int L3 = 2;


//hardware parameters
//PE array dimensions
#define P 1
//#define T 32
#define T 64 //original:128

#define P2 1
#define T2 2
//const int BLOCK_SIZE = 16;
// PE array sizes


typedef struct {
	float a[BSIZE];
} blockvec;

typedef struct {
	float a[L2/4]; //128 
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

void loadIn(float In[],  hls::stream<blockvec> &Inrows,const int LL, int it);
void loadW(float W[], w1blockvec w1bram[], const int LL, const int LN, int it);
void loadW2(float W[], w3blockvec w2bram[], const int LL, const int LN, int it);
void loadDDR(float In[], float W[], hls::stream<blockvec> &Arows, w1blockvec w1bram[], const int LL,const int LN,int it);
void blockmatmul(hls::stream<blockvec> &Inrows, hls::stream<blockvec> &Crows,w1blockvec Wcols[], const int LL,const int LN,int it);
void write_out_stream(float C[BSIZE/P][512/T][P][T], hls::stream<blockvec> &Crows,const int LN);
void blockmatmul3(hls::stream<blockvec> &Inrows, w3blockvec Wcols[], hls::stream<blockvec> &Crows,const int LL,const int LN);
void activation(hls::stream<blockvec> &Inrows, float bias[], hls::stream<blockvec> &Outrows,const int L);
void storeDDR(float O[],  hls::stream<blockvec> &Crows,  const int LN);
void top(float *A, float *B1,float *B2,float *O);
}