#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include "common.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "common.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif

std::string load_kernel_code(const std::string& kernel_path)
{
    std::cout << kernel_path << std::endl;
    std::ifstream f(kernel_path.c_str());
    std::stringstream buf;
    buf << f.rdbuf();

    return buf.str();
}

void img_info(const cv::Mat& img)
{
    std::cout << "Image size: " << img.rows << ", " << img.cols << std::endl;
    for(int r = 0; r < img.rows; r ++)
    {
        for(int c = 0; c < img.cols; c ++)
        {
            std::cout << std::hex << std::setw(8) << std::setfill('0') << img.at<int>(r, c) << ' ';
        }
        std::cout << std::endl;
    }
}
int main(int argc, char **argv)
{
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR_BGR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << argv[1] << std::endl;
        return 1;
    }
    img_info(img);
    return 0;
}