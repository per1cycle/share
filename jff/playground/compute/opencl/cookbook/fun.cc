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
    return 0;
}