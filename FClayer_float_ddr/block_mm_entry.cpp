#include "./block.h"

extern "C"{
void loadA(blockvec A[],  hls::stream<blockvec> &Arows, int it){
	int A_tile_index = int(it/(SIZE/BLOCK_SIZE));
	for (int i = 0; i < SIZE; i++){
		#pragma HLS PIPELINE
		Arows.write(A[A_tile_index*SIZE+i]);
	}
}

void loadB(blockvec B[], hls::stream<blockvec> &Bcols, int it){
	int B_tile_index = it%(SIZE/BLOCK_SIZE);
	for (int i = 0; i < SIZE; i++){
		#pragma HLS PIPELINE
		Bcols.write(B[i+SIZE*B_tile_index]);
	}
}

void loadDDR(blockvec A[], blockvec B[], hls::stream<blockvec> &Arows, hls::stream<blockvec> &Bcols, int it){
	// #pragma INTERFACE variable=A
	// #pragma INTERFACE variable=B
	//Assumption: A and B are entire matrices SIZE*BLOCK_SIZE(e.g. blockvec size) tiles
	#pragma HLS DATAFLOW
	loadA(A, Arows, it);
	loadB(B, Bcols, it);
}



void blockmatmul(hls::stream<blockvec> &Arows, hls::stream<blockvec> &Bcols, hls::stream<blockmat> &Cblocks, int it) {
#pragma HLS aggregate variable=Arows
#pragma HLS aggregate variable=Bcols
#pragma HLS aggregate variable=Cblocks
	float C[BLOCK_SIZE/P][BLOCK_SIZE/T][P][T]={0};
	#pragma HLS ARRAY_PARTITION variable=C dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=C dim=4 complete

	partialsum: for(int k=0; k < SIZE; k++) {
		blockvec tempA = Arows.read();
		blockvec tempB = Bcols.read();
    #pragma HLS aggregate variable=tempA
     #pragma HLS aggregate variable=tempB
		for(int i = 0; i < BLOCK_SIZE/P; i++) {
			for(int j = 0; j < BLOCK_SIZE/T; j++) {
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

	//write back
	blockmat tempC;
 #pragma HLS aggregate variable=tempC
	for(int i = 0; i < BLOCK_SIZE/P; i++) {
		#pragma HLS PIPELINE
		for(int j = 0; j < BLOCK_SIZE/T; j++) {
			for(int ii = 0; ii < P; ii++) {
				for(int jj = 0; jj < T; jj++) {
					//#pragma HLS dependence variable=C inter false
					tempC.out[i*P+ii][j*T+jj]=C[i][j][ii][jj];
				}
			}
		}
	}
	Cblocks.write(tempC);

	//==============below only uncomment for testbench===============
//	FILE *fp;
//	fp=fopen("./out.dat","w");
//	 for (int i=0; i<BLOCK_SIZE/P; i++){
//		 for (int j = 0; j < BLOCK_SIZE/T; j++) {
//			 for(int ii = 0; ii < P; ii++) {
//				 for(int jj = 0; jj < T; jj++) {
//					 fprintf(fp, "%f\n", C[i][j][ii][jj]);
//				 }
//			 }
//		 }
//	 }
//	 fclose(fp);
	 //==============above only uncomment for testbench===============
}

void storeDDR(blockmat C[],  hls::stream<blockmat> &Cblocks,  int it){
	// #pragma INTERFACE variable=A
	// #pragma INTERFACE variable=B
	//Assumption: A and B are entire matrices SIZE*BLOCK_SIZE(e.g. blockvec size) tiles
	//int A_tile_index = int(it/(SIZE/BLOCK_SIZE));
	//int B_tile_index = it%(SIZE/BLOCK_SIZE);
	C[it] = Cblocks.read();
}

void top(blockvec *A, blockvec *B, blockmat *C){
//#pragma HLS INTERFACE bram port=C storage_type=ram_2p
	//Put DDR interfacing directives for A & B
	#pragma HLS INTERFACE m_axi port=A bundle=gmem0 offset=slave
	#pragma HLS INTERFACE m_axi port=B bundle=gmem1 offset=slave
	#pragma HLS INTERFACE s_axilite port=A bundle=control
	#pragma HLS INTERFACE s_axilite port=B bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	
	//Assume C is buffered on-chip
//	blockmat C[SIZE*SIZE/(BLOCK_SIZE*BLOCK_SIZE)];
//#pragma HLS aggregate variable=C

	hls::stream<blockvec> pipe[2];
	hls::stream<blockmat> pipeout;
	#pragma HLS STREAM variable=pipe depth=4

	for (int it=0;it<SIZE*SIZE/(BLOCK_SIZE*BLOCK_SIZE);it++){
		#pragma HLS DATAFLOW
		loadDDR(A, B, pipe[0], pipe[1], it);
		blockmatmul(pipe[0], pipe[1],  pipeout, it);
		storeDDR(C, pipeout, it);
	}
}
}



