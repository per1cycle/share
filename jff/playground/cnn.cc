#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cassert>
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
    std::uint32_t Read4Byte()
    {
        assert(raw_data_.size() >= 4);
        std::uint8_t block0 = raw_data_[0];
        std::uint8_t block1 = raw_data_[1];
        std::uint8_t block2 = raw_data_[2];
        std::uint8_t block3 = raw_data_[3];
        raw_data_.erase(raw_data_.begin(), raw_data_.begin() + 4);
        return ((block0 << 24) | (block1 << 16) | (block2 << 8) | block3);
    }

    std::uint64_t Read8Byte()
    {
        assert(raw_data_.size() >= 8);
        for(int i = 0; i < 64; i ++)
        {
            std::cout << std::hex << static_cast<std::uint32_t>(raw_data_[i]) << ' ';
            if(i > 0 && i % 8 == 0)
            {
                std::cout << std::endl;
            }
        }
        std::uint8_t block0 = raw_data_[0];
        std::uint8_t block1 = raw_data_[1];
        std::uint8_t block2 = raw_data_[2];
        std::uint8_t block3 = raw_data_[3];
        std::uint8_t block4 = raw_data_[4];
        std::uint8_t block5 = raw_data_[5];
        std::uint8_t block6 = raw_data_[6];
        std::uint8_t block7 = raw_data_[7];
        // std::cout << "Block 7 " << std::hex << static_cast<std::uint32_t>(block7) << std::endl;
        raw_data_.erase(raw_data_.begin(), raw_data_.begin() + 8);

        return 
            (
              (static_cast<std::uint64_t>(block0) << 56)
            | (static_cast<std::uint64_t>(block1) << 48)
            | (static_cast<std::uint64_t>(block2) << 40) 
            | (static_cast<std::uint64_t>(block3) << 32) 
            | (static_cast<std::uint64_t>(block4) << 24) 
            | (static_cast<std::uint64_t>(block5) << 16) 
            | (static_cast<std::uint64_t>(block6) << 8) 
            | (static_cast<std::uint64_t>(block7) << 0)
            );
    }

    int Resolve()
    {
        // resolve a png data.
        std::uint64_t Header = Read8Byte();
        if (Header != 0x89504e470d0a1a0a)
        {
            std::cerr << std::hex << Header << " Format error. \n";
            return 1;
        }
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
    image.Resolve();
    return 0;
}