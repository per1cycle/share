#include <OpenCL/cl.h>
#include <iostream>

std::string kernel_code = 
"__kernel" \
"{" \
"}"
;

int main()
{
    std::cout << kernel_code << std::endl;

    return 0;
}