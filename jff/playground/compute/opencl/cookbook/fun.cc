#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>

#include "common.h"
#include "mycl.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif

int main(int argc, char** argv)
{
    mycl::list_platform_and_device();
    int row = 3, col = 5;
    auto res = mycl::generate_NxM_image(row, col);
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < col; j ++)
        {
            std::cout << res[i * col + j] << ' ';
        }
        std::cout << std::endl;
    }    
    return 0;
}