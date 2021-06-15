#include "./block.h"

extern "C"{
//input tile: LL=L1
void loadIn(float In[],  hls::stream<blockvec> &Inrows,const int LL, int it){
	#pragma HLS aggregate variable=Inrows
//	int A_tile_index = int(it/(BSIZE/BLOCK_SIZE));
	for (int i = 0; i < LL; i++){
     	blockvec tempA;
     	#pragma HLS aggregate variable=tempA
     	for (int j = 0; j < BSIZE; j++){
			#pragma HLS PIPELINE
			tempA.a[j]=In[it*LL*BSIZE+i*BSIZE+j];
	  	}
   		Inrows.write(tempA);
	}
}


//weight tile: LL=L1/L2/L3
void loadW(float W[], w1blockvec w1bram[], const int LL, const int LN, int it){
	#pragma HLS aggregate variable=w1bram
//	int B_tile_index = it%(BSIZE/BLOCK_SIZE);
	for (int i = 0; i < LL; i++){
		w1blockvec tempA;
		#pragma HLS aggregate variable=tempA
		for (int j = 0; j < LN; j++){
			#pragma HLS PIPELINE
			tempA.a[j]=W[it*LL*LN+i*LN+j];
		}
		// Wcols.write(tempA)
		w1bram[i]=tempA;
	}
}


//weight tile: LL=L1/L2/L3
void loadW2(float W[], w3blockvec w2bram[], const int LL, const int LN, int it){
//	int B_tile_index = it%(BSIZE/BLOCK_SIZE);
	for (int i = 0; i < LL; i++){
		w3blockvec tempA;
		#pragma HLS aggregate variable=tempA
		for (int j = 0; j < LN; j++){
			#pragma HLS PIPELINE
			tempA.a[j]=W[it*LL*LN+i*LN+j];
		}
		// Wcols.write(tempA)
		w2bram[i]=tempA;
	}
}


void loadDDR(float In[], float W[], hls::stream<blockvec> &Arows, w1blockvec w1bram[], const int LL,const int LN,int it){
	// #pragma INTERFACE variable=A
	// #pragma INTERFACE variable=B
	//Assumption: A and B are entire matrices SIZE*BLOCK_SIZE(e.g. blockvec size) tiles
	#pragma HLS DATAFLOW
	loadIn(In, Arows, LL,it);
	loadW(W, w1bram,LL,LN, it);
}


//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
void blockmatmul(hls::stream<blockvec> &Inrows, w1blockvec Wcols[], float C[BSIZE/P][512/T][P][T], const int LL,const int LN) {
#pragma HLS aggregate variable=Inrows
#pragma HLS aggregate variable=Wcols
#pragma HLS aggregate variable=Crows
	// float C[BSIZE/P][512/T][P][T]={0}; //512 is the largest layer. change based on models
	// #pragma HLS ARRAY_PARTITION variable=C dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=C dim=4 complete

	partialsum: for(int k=0; k < LL; k++) {
		blockvec tempA = Inrows.read();
		w1blockvec tempB = Wcols[k];
    #pragma HLS aggregate variable=tempA
     #pragma HLS aggregate variable=tempB
		for(int i = 0; i < BSIZE/P; i++) {
			for(int j = 0; j < LN/T; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=C inter false
				for(int ii = 0; ii < P; ii++) {
					#pragma HLS UNROLL
					for(int jj = 0; jj < T; jj++) {
						#pragma HLS UNROLL
						//#pragma HLS dependence variable=C inter false
						C[i][j][ii][jj] = C[i][j][ii][jj] + tempA.a[i*P+ii] * tempB.a[j*T+jj];
					}
				}
			}
		}
	}
	//write out to stream

}

void write_out_stream(float C[BSIZE/P][512/T][P][T], hls::stream<blockvec> &Crows,const int LN){

	for(int j = 0; j < LN/T; j++) {
		for(int jj = 0; jj < T; jj++) {
     #pragma HLS PIPELINE
			blockvec tempC;
			#pragma HLS aggregate variable=tempC
			for(int i = 0; i < BSIZE/P; i++) {
				for(int ii = 0; ii < P; ii++) {
					tempC.a[i*P+ii]=C[i][j][ii][jj];
				}
			}
			Crows.write(tempC);
		}
	}
}


//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
void blockmatmul3(hls::stream<blockvec> &Inrows, w3blockvec Wcols[], hls::stream<blockvec> &Crows,const int LL,const int LN) {
#pragma HLS aggregate variable=Inrows
#pragma HLS aggregate variable=Wcols
#pragma HLS aggregate variable=Crows
	float C[BSIZE/P][2/T2][P2][T2]={0};
	#pragma HLS ARRAY_PARTITION variable=C dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=C dim=4 complete

	partialsum: for(int k=0; k < LL; k++) {
		blockvec tempA = Inrows.read();
		w3blockvec tempB = Wcols[k];
    #pragma HLS aggregate variable=tempA
     #pragma HLS aggregate variable=tempB
		for(int i = 0; i < BSIZE/P2; i++) {
			for(int j = 0; j < LN/T2; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=C inter false
				for(int ii = 0; ii < P2; ii++) {
					#pragma HLS UNROLL
					for(int jj = 0; jj < T2; jj++) { //3
						#pragma HLS UNROLL
						C[i][j][ii][jj] = C[i][j][ii][jj] + tempA.a[i*P2+ii] * tempB.a[j*T2+jj];
					}
				}
			}
		}
	}
	//write out to stream
	for(int j = 0; j < LN/T2; j++) {
		for(int jj = 0; jj < T2; jj++) {
   			#pragma HLS PIPELINE
			blockvec tempC;
			#pragma HLS aggregate variable=tempC
			for(int i = 0; i < BSIZE/P2; i++) {
				for(int ii = 0; ii < P2; ii++) {
					tempC.a[i*P2+ii]=C[i][j][ii][jj];
				}
			}
			Crows.write(tempC);
		}
	}
}

void activation(hls::stream<blockvec> &Inrows, float bias[], hls::stream<blockvec> &Outrows,const int L){
	for (int i = 0; i < L; i++){
		#pragma HLS PIPELINE
		blockvec temp = Inrows.read();
		blockvec temp_out;
		for (int j = 0; j < BSIZE; j++){
			#pragma HLS UNROLL
			temp_out.a[j]=hls::tanh(temp.a[j]+bias[i]);
			// temp.a[j]=tmp;
		}
		Outrows.write(temp_out);
	}
}
void storeDDR(float O[],  hls::stream<blockvec> &Crows,  const int LN){
	for (int i = 0; i < LN; i++){
   //printf("In itr %d\n",i);
		blockvec tmp = Crows.read();
     for (int j = 0; j < BSIZE; j++){
     #pragma HLS PIPELINE
       O[i*BSIZE+j]=tmp.a[j];
     }
	}
 //printf("Yaaassss\n");

}

void top(float *A, float *B1,float *B2,float *O){
//#pragma HLS INTERFACE bram port=C storage_type=ram_2p
	//Put DDR interfacing directives for A & B
	#pragma HLS INTERFACE m_axi port=A bundle=gmem0 offset=slave
	#pragma HLS INTERFACE m_axi port=B1 bundle=gmem1 offset=slave
	#pragma HLS INTERFACE m_axi port=B2 bundle=gmem2 offset=slave
	#pragma HLS INTERFACE m_axi port=O bundle=gmem3 offset=slave
	#pragma HLS INTERFACE s_axilite port=A bundle=control
	#pragma HLS INTERFACE s_axilite port=B1 bundle=control
	#pragma HLS INTERFACE s_axilite port=B2 bundle=control
	#pragma HLS INTERFACE s_axilite port=O bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	
	//Assume C is buffered on-chip
//	blockmat C[SIZE*SIZE/(BLOCK_SIZE*BLOCK_SIZE)];
//#pragma HLS aggregate variable=C

	hls::stream<blockvec> inpipe;
	w1blockvec w1bram[L1];
	w3blockvec w2bram[L2];

  float bias1[L2]={0};
  float bias2[L3]={0.46203604340553284,0.37419500946998596};
	
	float C1[BSIZE/P][512/T][P][T]={0}; //512 is the largest layer. change based on models
	#pragma HLS ARRAY_PARTITION variable=C1 dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=C1 dim=4 complete


	hls::stream<blockvec> outpipe[4];
	//hls::stream<blockvec> actpipe[3];
	#pragma HLS STREAM variable=inpipe depth=64
	#pragma HLS STREAM variable=outpipe depth=64

//	Init on-chip memory

{	
  #pragma HLS DATAFLOW
	for (int it=0; it<L1_total/L1;it++){
		// need a iteration number in the argument! - it
		loadDDR(A, B1, inpipe, w1bram, L1,L2,it);
		blockmatmul(inpipe, w1bram, C1, L1,L2);
	}
	write_out_stream(C1, outpipe[0],L2);
 //printf("load 1\n");
//	loadW(w1blockvec W[], wpipe[1], L1);
	
 //printf("MM 1\n");
//	blockmatmul(outpipe[0], w2bram, outpipe[1],L2,L3);
// printf("MM 2\n");
  activation(outpipe[0], bias1, outpipe[1],L2);
  //printf("activation 1\n");
  loadW2(B2, w2bram, L2,L3,0);
	blockmatmul3(outpipe[1], w2bram, outpipe[2],L2,L3);
  //printf("MM 2\n");
  activation(outpipe[2], bias2, outpipe[3],L3);
  //printf("activation 2\n");
	storeDDR(O, outpipe[3], L3);
  //printf("kernel really finished\n");
}
}
}



