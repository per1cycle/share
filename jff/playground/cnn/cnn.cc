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

        // std::cout << "Read data speed: " 
        //     << raw_data_.size() * 1.0 / elapsed.count() 
        //     << " KByte/s" << std::endl;

        // std::cout << "Load image in: "
        //     << elapsed.count() << " milliseconds." 
        //     << std::endl;
    }

public:
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

    void PrintImage()
    {
        for(int i = 0; i < img_height_; i ++)
        {
            for(int j = 0; j < img_width_; j ++)
            {
                std::cout << std::hex << std::setw(8) << std::setfill('0') << data_[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

   std::uint32_t Read4Byte(std::vector<std::uint8_t>& v)
    {
        assert(v.size() >= 4);
        std::uint8_t block0 = v[0];
        std::uint8_t block1 = v[1];
        std::uint8_t block2 = v[2];
        std::uint8_t block3 = v[3];
        v.erase(v.begin(), v.begin() + 4);
        return ((block0 << 24) | (block1 << 16) | (block2 << 8) | block3);
    }
    
    std::uint64_t Read8Byte(std::vector<std::uint8_t>& v)
    {
        assert(v.size() >= 8);
        
        std::uint8_t block0 = v[0];
        std::uint8_t block1 = v[1];
        std::uint8_t block2 = v[2];
        std::uint8_t block3 = v[3];
        std::uint8_t block4 = v[4];
        std::uint8_t block5 = v[5];
        std::uint8_t block6 = v[6];
        std::uint8_t block7 = v[7];

        v.erase(v.begin(), v.begin() + 8);

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

    std::vector<std::uint8_t> ReadNByte(std::vector<std::uint8_t>& v, std::uint32_t n)
    {
        assert(v.size() > n);
        std::vector<std::uint8_t> ret(v.begin(), v.begin() + n);
        v.erase(v.begin(), v.begin() + n);
        return ret;
    }

    std::vector<std::uint32_t> ResolveOneLine(std::vector<std::uint8_t>& raw_data, std::uint32_t length)
    {
        assert((length % 4) == 1); // image width + crc
        std::vector<std::uint32_t> result;
        result.resize(length / 4);
        for(int i = 0; i < length / 4; i ++)
        {
            std::uint32_t data = Read4Byte(raw_data);
            result[i] = data;
        }
        raw_data.erase(raw_data.begin());
        return result;
    }

    png_chunk* ReadChunk()
    {
        png_chunk *chunk = new png_chunk;
        chunk->length = Read4Byte(raw_data_);
        // std::cout << "Read chun length: " << chunk->length << std::endl;

        chunk->chunk_type.value = Read4Byte(raw_data_);
        // std::cout << "Chunk type: " << 
        //     chunk->chunk_type.type_name[3] << 
        //     chunk->chunk_type.type_name[2] << 
        //     chunk->chunk_type.type_name[1] << 
        //     chunk->chunk_type.type_name[0] << 
        //     ". Chunk length: " << 
        //     chunk->length << 
        //     ". Type in hex: " << 
        //     std::hex << chunk->chunk_type.value << 
        //     std::endl;
        
        // https://www.w3.org/TR/2003/REC-PNG-20031110/#5Chunk-layout
        chunk->chunk_data.resize(chunk->length);
        chunk->chunk_data = ReadNByte(raw_data_, chunk->length);
        chunk->crc = Read4Byte(raw_data_);

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

    void DumpIDATData(const png_chunk &chunk)
    {
        std::ofstream of("idat.out");
        std::ostream_iterator<std::uint8_t> it(of);
        std::copy(chunk.chunk_data.begin(), chunk.chunk_data.end(), it);
    }

    /**
     * Decompress a sequance of byte to a image data.
     * 2025/03/27 implementation:
     * Give up, and use zlib default impl
     * 
     */
    std::vector<std::uint8_t> Decompress(std::vector<std::uint8_t>& data)
    {
        // compress example
        // std::vector<uint8_t> decompressed_data;
        // decompressed_data.resize(img_width_ * img_height_ * 4);
        uLong ucomp_size = img_width_ * img_height_ * 4 + img_height_;
        uLong comp_size = data.size();
        std::vector<std::uint8_t> temp(ucomp_size);
        int result = uncompress(static_cast<Bytef*>(&temp[0]), &ucomp_size, &data[0], comp_size);

        if(result == Z_OK)
        {
            return temp;
        }
        return {0};
    }
    
public:
    int Resolve()
    {
        // resolve a png data.
        std::uint64_t Header = Read8Byte(raw_data_);
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
            {
                img_width_ = Read4Byte(chunk->chunk_data);
                img_height_ = Read4Byte(chunk->chunk_data);
                data_.resize(img_height_);
                for(int i = 0; i < img_height_; i ++)
                {
                    data_[i].resize(img_width_ + 1);
                }
                break;
            }
            case PHYS:
                break;
            case IDAT:
            {
                std::vector<std::uint8_t> ret = Decompress(chunk->chunk_data);
                std::cout << "Image return size: " << std::dec << ret.size() << std::endl;
                std::uint32_t line = img_height_;
                std::uint32_t byte_per_line = img_width_ * 4 + 1;
                for(int i = 0; i < line; i ++)
                {
                    std::vector<std::uint32_t> temp;
                    temp = ResolveOneLine(ret, byte_per_line);
                    data_[i] = temp;
                }
                assert(ret.size() == 0);
                PrintImage();
                break;
            }
            case IEND:
                break;
            }
            total_size += chunk->length;
        }

        // std::cout << std::dec << "Image data: " << total_size << std::endl;
        return 0;
    }

    int CNN()
    {
        
        return 0;
    }

private:
    std::vector<std::uint8_t> raw_data_;
    std::vector<std::vector<std::uint32_t> > data_;
    std::vector<std::vector<int> > result_;
    std::vector<std::vector<int> > core_;
    std::uint32_t img_width_;
    std::uint32_t img_height_;
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