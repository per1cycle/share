#include <stdio.h>
void print_binary(unsigned char ch)
{
    for(int i = 0; i < 8; i ++)
    {
        printf("%c", ((ch >> i) & 0x1)? '1' : '0'); 
    }
}

void print_float(float f)
{
    unsigned char* c = &f;
    // notice the system is small endian.
    for(int i = 0; i < 4; i ++)
    {
        print_binary(c[i]);
    }
    puts("\n");
}

int main()
{
    float f = 1.0f;
    print_float(f);
    return 0;
}