#include "mycl.h"

int main()
{
    cl_device_id device_id = mycl::select_device(0, 0);
    
    cl_context context = mycl::simple::simple_create_context(&device_id);
    
}