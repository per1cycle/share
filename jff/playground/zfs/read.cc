#include "zfs.h"
#include <iostream>

void better_usage()
{
    std::cerr << "Usage: \n" << 
        "Subcommand: [ls|cat|cp]";
    exit(1);
}

void usage()
{
    std::cerr << "Usage: " << 
        "./zread <path to zfs img>" << 
        std::endl;
    exit(1);
}

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        usage();
    }

    return 0;
}