#include <ctype.h>
#include <stdio.h>

/*
global variable
- file
- token type
*/
struct token;

FILE* input_file;
int line_number = 1;
char replace_char = 0;
struct token g_token;

const char* tokens[] = {
    "+",
    "-",
    "*",
    "/",
    "integer"
};

enum TOKEN_TYPE {
    PLUS_T = 0,
    MINUS_T,
    MUL_T,
    DIV_T,
    INT_T
};

struct token 
{
    enum TOKEN_TYPE type;
    int value;
};

void usage()
{
    printf("Usage: ./c <input file> \n");
}

int next()
{
    int c;
    if(replace_char)
    {
        c = replace_char;
        replace_char = 0;
        return c;
    }

    c = fgetc(input_file);
    if(c == '\n') // if goto next line.
        line_number ++;
    return c;

}

int skip()
{
    int c = next();
    while(c == ' ' || c == '\n' || c == '\t' || c == '\r')
        c = next();
    return c;
}

int scan()
{
    int c;
    c = skip();
    switch (c) 
    {
        case EOF:
            return 0;
        case '+':
        {
            printf("Found add\n");
            g_token.type = PLUS_T;
            break;
        }
        case '-':
        {
            printf("Found minus\n");

            g_token.type = MINUS_T;
            break;
        }
        case '*':
        {
            printf("Found mul\n");
            g_token.type = MUL_T;
            break;
        }
        case '/':
        {
            printf("Found div\n");
            g_token.type = DIV_T;
            break;
        }
        default:
        {
            // will replace with next_digit() here.
            if (isdigit(c))
            {
                printf("Find digits.\n");
                break;
            }
        }
            
    }
    return c;
}

void scan_file()
{
    while (scan())
    {
        
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        usage();
        return 1;
    }
    char* file_name = argv[1];
    input_file = fopen(file_name, "r");

    scan_file();

    return 0;
}