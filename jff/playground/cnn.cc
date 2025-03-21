#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <iterator>

/**
 * conv matrix
 */
class Mat
{
public:
    Mat(std::string filename)
    {
        // read a image
        auto start = std::chrono::high_resolution_clock::now();
        std::ifstream is(filename.c_str(), std::ios::binary);
        std::vector<std::uint8_t> temp(
            (std::istream_iterator<std::uint8_t>(is)),
            std::istream_iterator<std::uint8_t>()
        );
        raw_data_.resize(temp.size());
        std::copy(temp.begin(), temp.end(), raw_data_.begin());
        std::cout << "Raw data read " << raw_data_.size() << " bytes" 
                    << std::endl;
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = finish - start;
        std::cout << "Read data speed: " 
            << raw_data_.size() * 1.0 / elapsed.count() 
            << " KByte/s" << std::endl;
        std::cout << "Read data speed: "
            << elapsed.count() << " milliseconds." 
            << std::endl;
    }

public:
    int Resolve()
    {
        // resolve a png data.
        
        return 0;
    }

    int CNN()
    {
        
        return 0;
    }

private:
    std::vector<std::uint8_t> raw_data_;
    std::vector<std::vector<std::uint32_t>> data_;
    std::vector<std::vector<int>> result_;
    std::vector<std::vector<int>> core_;
    int stride_ = 1;
    bool padding_ = false;
};

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        std::cerr << "Usage: ./cnn <image path>.png" << std::endl;
        exit(1);
    }

    Mat image(argv[1]);
    
    return 0;
}