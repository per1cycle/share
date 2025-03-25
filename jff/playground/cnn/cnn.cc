#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cassert>
#include <iterator>
#include <zlib.h>

#define PNG_HDR 0x89504e470d0a1a0a
#define IHDR 0x49484452
#define PHYS 0x70485973
#define IDAT 0x49444154
#define IEND 0x49454e44

typedef struct pngchunk
{
    std::uint32_t length;
    union {
        std::uint32_t value;
        char type_name[4];
    } chunk_type;
    std::vector<std::uint8_t> chunk_data;
    std::uint32_t crc;
} png_chunk;

/**
 * conv matrix
 */
class Mat
{
public:
    Mat(std::string filename)
    {
        // read a image
        auto start = std::chrono::high_resolution_clock::now(); // start timer

        std::ifstream is(filename.c_str(), std::ifstream::binary);

        // read file
        is.seekg(0, std::ios::end);
        size_t file_size = is.tellg();
        is.seekg(0, std::ios::beg);
        
        raw_data_.resize(file_size);
        is.read(reinterpret_cast<char*>(raw_data_.data()), file_size);

        auto finish = std::chrono::high_resolution_clock::now(); // end timer
        std::chrono::duration<double, std::milli> elapsed = finish - start;

        std::cout << "Read data speed: " 
            << raw_data_.size() * 1.0 / elapsed.count() 
            << " KByte/s" << std::endl;

        std::cout << "Load image in: "
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

    void PrintRaw()
    {
        for(int i = 0; i < 64; i ++)
        {
            std::cout << std::hex << static_cast<std::uint32_t>(raw_data_[i]) << ' ';
            if(i > 0 && (i + 1) % 8 == 0)
            {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    std::uint64_t Read8Byte()
    {
        assert(raw_data_.size() >= 8);
        
        std::uint8_t block0 = raw_data_[0];
        std::uint8_t block1 = raw_data_[1];
        std::uint8_t block2 = raw_data_[2];
        std::uint8_t block3 = raw_data_[3];
        std::uint8_t block4 = raw_data_[4];
        std::uint8_t block5 = raw_data_[5];
        std::uint8_t block6 = raw_data_[6];
        std::uint8_t block7 = raw_data_[7];

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

    std::vector<std::uint8_t> ReadNByte(std::uint32_t n)
    {
        assert(raw_data_.size() > n);
        std::vector<std::uint8_t> ret(raw_data_.begin(), raw_data_.begin() + n);
        raw_data_.erase(raw_data_.begin(), raw_data_.begin() + n);
        return ret;
    }
 
    png_chunk* ReadChunk()
    {
        png_chunk *chunk = new png_chunk;
        chunk->length = Read4Byte();
        // std::cout << "Read chun length: " << chunk->length << std::endl;

        chunk->chunk_type.value = Read4Byte();
        std::cout << "Chunk type: " << 
            chunk->chunk_type.type_name[3] << 
            chunk->chunk_type.type_name[2] << 
            chunk->chunk_type.type_name[1] << 
            chunk->chunk_type.type_name[0] << 
            ". Chunk length: " << 
            chunk->length << 
            ". Type in hex: " << 
            std::hex << chunk->chunk_type.value << 
            std::endl;
        
        // https://www.w3.org/TR/2003/REC-PNG-20031110/#5Chunk-layout
        chunk->chunk_data.resize(chunk->length);
        chunk->chunk_data = ReadNByte(chunk->length);
        chunk->crc = Read4Byte();

        if(chunk->length == 0 && chunk->chunk_type.value == 0x49454e44)
        {
            delete chunk;
            return nullptr;
        }

        return chunk;
    }
   
    /**
     * calculate crc of current chunk
     * todo, not planned in current version.
     */
    std::uint32_t Crc(std::uint32_t chunk_type, const std::vector<int> &data)
    {
        std::uint32_t crc_register = 1;
        return 0;
    }

    /**
     * Decompress a sequance of byte to a image data.
     * algorithm: lz77
     */
    std::vector<std::uint8_t> Decompress(const std::vector<std::uint8_t>& data)
    {

    }
    
public:
    int Resolve()
    {
        // resolve a png data.
        std::uint64_t Header = Read8Byte();
        if (Header != PNG_HDR)
        {
            std::cerr << std::hex << Header << " Format error. \n";
            return 1;
        }

        png_chunk *chunk;
        std::uint32_t total_size = 0;

        while((chunk = ReadChunk()) != nullptr)
        {
            switch (chunk->chunk_type.value)
            {
            case IHDR:
                break;
            case PHYS:
                break;
            case IDAT:
                for(int i = 0; i < chunk->length; i ++)
                {
                    if(i > 0 && i % 8 == 0)
                    {
                        std::cout << std::endl;
                    }
                    std::cout << std::hex << std::setfill('0')<< std::setw(2) << static_cast<std::uint16_t>(chunk->chunk_data[i]) << ' ';
                }
                std::cout << std::endl;
                break;
            case IEND:
                break;
            }
            total_size += chunk->length;
        }

        std::cout << std::dec << "Image data: " << total_size << std::endl;
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