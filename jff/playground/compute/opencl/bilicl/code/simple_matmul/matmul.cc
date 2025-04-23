#include "mycl.h"

int main(int argc, char ** argv)
{
    cl_device_id device_id = mycl::select_device(0, 0);
    
    cl_context ctx = mycl::simple::simple_create_context(&device_id, 0);
    
    cl_command_queue cq = mycl::simple::simple_create_command_queue(ctx, device_id);

    int N, M, K;
    N = 1024, M = 1024, K = 1024;

    float *h_a = new float[N * M];
    float *h_b = new float[M * K];
    float *h_c = new float[N * K];

    mycl::generate_NxM_mat(h_a, N, M);
    mycl::generate_NxM_mat(h_b, M, K);
    mycl::generate_NxM_mat(h_c, N, K);
    
    cl_program pgm = mycl::simple::simple_create_program_and_build();

    return 0;
}