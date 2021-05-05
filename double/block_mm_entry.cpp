#include "./block.h"

extern "C"{
//input tile: LL=L1
void loadIn(blockvec In[],  hls::stream<blockvec> &Inrows,const int LL){
//	int A_tile_index = int(it/(BSIZE/BLOCK_SIZE));
	for (int i = 0; i < LL; i++){
		#pragma HLS PIPELINE
		Inrows.write(In[i]);
	}
}

////weight tile: LL=L1/L2/L3
//void loadW(w1blockvec W[], hls::stream<w1blockvec> &Wcols, int LL){
////	int B_tile_index = it%(BSIZE/BLOCK_SIZE);
//	for (int i = 0; i < LL; i++){
//		#pragma HLS PIPELINE
//		Wcols.write(W[i]);
//	}
//}


//void loadDDR(blockvec In[], blockvec B[], hls::stream<blockvec> &Arows, hls::stream<blockvec> &Bcols, int it){
//	// #pragma INTERFACE variable=A
//	// #pragma INTERFACE variable=B
//	//Assumption: A and B are entire matrices SIZE*BLOCK_SIZE(e.g. blockvec size) tiles
//	#pragma HLS DATAFLOW
//	loadIn(A, Arows, it);
//	loadB(B, Bcols, it);
//}


//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
void blockmatmul(hls::stream<blockvec> &Inrows, w1blockvec Wcols[], hls::stream<blockvec> &Crows, const int LL,const int LN) {
#pragma HLS aggregate variable=Inrows
#pragma HLS aggregate variable=Wcols
#pragma HLS aggregate variable=Crows
	double C[BSIZE/P][64/T][P][T]={0}; //64 is the largest layer. change based on models
	#pragma HLS ARRAY_PARTITION variable=C dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=C dim=4 complete

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
	double C[BSIZE/P][3/T2][P2][T2]={0};
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

void storeDDR(blockvec C[],  hls::stream<blockvec> &Crows,  const int LN){
	for (int i = 0; i < LN; i++){
		#pragma HLS PIPELINE
   printf("In itr %d\n",i);
		C[i] = Crows.read();
	}
 printf("Yaaassss\n");

}

void top(blockvec *A, blockvec *C){
//#pragma HLS INTERFACE bram port=C storage_type=ram_2p
	//Put DDR interfacing directives for A & B
	#pragma HLS INTERFACE m_axi port=A bundle=gmem0 offset=slave
	#pragma HLS INTERFACE m_axi port=C bundle=gmem1 offset=slave
	#pragma HLS INTERFACE s_axilite port=A bundle=control
	#pragma HLS INTERFACE s_axilite port=C bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	
	//Assume C is buffered on-chip
//	blockmat C[SIZE*SIZE/(BLOCK_SIZE*BLOCK_SIZE)];
//#pragma HLS aggregate variable=C

	hls::stream<blockvec> inpipe;
	w1blockvec w1bram[L1];
	w1blockvec w2bram[L2];
	w3blockvec w3bram[L3];
	hls::stream<blockvec> outpipe[3];
	#pragma HLS STREAM variable=inpipe depth=128
	#pragma HLS STREAM variable=outpipe depth=128
printf("everything good 1\n");
//	Init on-chip memory
	for (int i=0; i<L1;i++){
		#pragma HLS PIPELINE
		for  (int j=0; j<L2;j++){
			#pragma HLS UNROLL
			w1bram[i].a[j]=i;
		}
	}
	for (int i=0; i<L2;i++){
	#pragma HLS PIPELINE
		for  (int j=0; j<L3;j++){
			#pragma HLS UNROLL
			w2bram[i].a[j]=i;
		}
	}
	for (int i=0; i<L3;i++){
		#pragma HLS PIPELINE
		for  (int j=0; j<L4;j++){
			#pragma HLS UNROLL
			w3bram[i].a[j]=1;
		}
	}
 printf("initiation suceeded 1\n");
	
  #pragma HLS DATAFLOW
	loadIn(A, inpipe, L1);
 printf("load 1\n");
//	loadW(w1blockvec W[], wpipe[1], L1);
	blockmatmul(inpipe, w1bram, outpipe[0], L1,L2);
 printf("MM 1\n");
	blockmatmul(outpipe[0], w2bram, outpipe[1],L2,L3);
 printf("MM 2\n");
	blockmatmul3(outpipe[1], w3bram, outpipe[2],L3,L4);
 printf("MM 3\n");
	storeDDR(C, outpipe[2], L4);
  printf("kernel really finished\n");
}
}



