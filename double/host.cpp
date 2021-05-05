#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define DATA_SIZE 16
#define OUT_SIZE 4

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include "./block.h"

// Forward declaration of utility functions included at the end of this file
std::vector<cl::Device> get_xilinx_devices();
char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb);

// HBM Pseudo-channel(PC) requirements
#define MAX_HBM_PC_COUNT 16
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
    PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15)};


// ------------------------------------------------------------------------------------
// Main program
// ------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // ------------------------------------------------------------------------------------
    // Step 1: Initialize the OpenCL environment
    // ------------------------------------------------------------------------------------
    cl_int err;
    std::string binaryFile = (argc != 2) ? "top.xclbin" : argv[1];
    unsigned fileBufSize;
    std::vector<cl::Device> devices = get_xilinx_devices();
    devices.resize(1);
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    char *fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    cl::Program program(context, devices, bins, NULL, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl::Kernel krnl_vector_add(program, "top", &err);

    // ------------------------------------------------------------------------------------
    // Step 2: Create buffers and initialize test values
    // ------------------------------------------------------------------------------------
    //DATA_SIZE=BATCH SIZE
    std::vector<blockvec> In_rows(DATA_SIZE);
    std::vector<blockvec> C_rows(DATA_SIZE);

    printf("here 1\n");
    

    double A_ini[BSIZE*L1];
//    float C_ini[BSIZE*L4];
 //   float C_ref[BSIZE*L4];
    
    int i, j, k;
    std::cout << "init A_ini matrix." << std::endl;
    for (i = 0; i < BSIZE; i++) {
        for (j = 0; j < L1; j++) {
            A_ini[i*L1+j] = i+2;
        }
    }
    for (i = 0; i < BSIZE; i++) {
        for (j = 0; j < L1; j++) {
            In_rows[j].a[i] = A_ini[i*L1+j];

        }
    }

  printf("inied\n");
    
    cl_mem_ext_ptr_t InrExt;
    cl_mem_ext_ptr_t CrExt;
    

    InrExt.obj = In_rows.data();
    InrExt.param = 0;
    InrExt.flags = 1|XCL_MEM_TOPOLOGY;

    CrExt.obj = C_rows.data();
    CrExt.param = 0;
    CrExt.flags = 2|XCL_MEM_TOPOLOGY;

  printf("flags set\n");
    // Create the buffers and allocate memory
    cl::Buffer in1_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * DATA_SIZE, &InrExt, &err);
    cl::Buffer out_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * DATA_SIZE, &CrExt, &err);
      printf("hi\n");
    // Set kernel arguments
    krnl_vector_add.setArg(0, in1_buf);
    krnl_vector_add.setArg(1, out_buf);

    // Map host-side buffer memory to user-space pointers [replaced, used equeueMapBuffer]
    //blockvec *A = (blockvec *)q.enqueueMapBuffer(in1_buf, CL_TRUE, CL_MAP_WRITE, 0, sizeof(blockvec) * DATA_SIZE);
    //blockvec *B = (blockvec *)q.enqueueMapBuffer(in2_buf, CL_TRUE, CL_MAP_WRITE, 0, sizeof(blockvec) * DATA_SIZE);
    //blockmat *C = (blockmat *)q.enqueueMapBuffer(out_buf, CL_TRUE, CL_MAP_WRITE, 0, sizeof(blockmat) * OUT_SIZE);
    //std::vector<blockvec> A(DATA_SIZE);
    //std::vector<blockvec> B(DATA_SIZE);
    //std::vector<blockmat> C(OUT_SIZE);
    
      printf("setArg finished\n");

    FILE *fp3;
    fp3=fopen("./IOnkernel.dat","w");

    for (j = 0; j < L1; j++) {
        for (i = 0; i < BSIZE; i++) {
            fprintf(fp3,"%f ",In_rows[j].a[i]);
        }
        fprintf(fp3,"\n");
    }

    fclose(fp3);
    printf("starting kernel\n");
    // ------------------------------------------------------------------------------------
    // Step 3: Run the kernel
    // ------------------------------------------------------------------------------------

    krnl_vector_add.setArg(0, in1_buf);
    krnl_vector_add.setArg(1, out_buf);
    printf("setArg\n");
    // Schedule transfer of inputs to device memory, execution of kernel, and transfer of outputs back to host memory
    q.enqueueMigrateMemObjects({in1_buf}, 0 /* 0 means from host*/);
    printf("sent data\n");
    q.enqueueTask(krnl_vector_add);
    q.finish();
    printf("executed kernel\n");
    q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    printf("data back\n");

    // Wait for all scheduled operations to finish
    q.finish();
    printf("q.finish\n");
    // ------------------------------------------------------------------------------------
    // Step 4: Check Results and Release Allocated Resources
    // ------------------------------------------------------------------------------------
    bool match = true;
    printf("hi\n");
    FILE *fp;
    fp=fopen("./Crows.dat","w");
    printf("hi\n");
    for (j = 0; j < L4; j++) {
        for (i = 0; i < BSIZE; i++) {
            fprintf(fp, "%f ", C_rows[j].a[i]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
 
    delete[] fileBuf;

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

// ------------------------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------------------------
std::vector<cl::Device> get_xilinx_devices()
{
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++)
    {
        platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == "Xilinx")
        {
            std::cout << "INFO: Found Xilinx Platform" << std::endl;
            break;
        }
    }
    if (i == platforms.size())
    {
        std::cout << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }

    //Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}

char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb)
{
    if (access(xclbin_file_name.c_str(), R_OK) != 0)
    {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    std::cout << "INFO: Loading '" << xclbin_file_name << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char *buf = new char[nb];
    bin_file.read(buf, nb);
    return buf;
}