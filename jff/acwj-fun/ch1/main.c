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
int g_parse_digit = 0;

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

void set_replace_char(int c)
{
    replace_char = c;
}

void parse_panic(const char* panic_info)
{
    printf("[ERROR]: %s\n", panic_info);
    for(;;);
}

int next_digit(int start_digit)
{
    int result = start_digit - '0';
    int c;
    while((c = next()) && isdigit(c))
    {
        result = result * 10 + c - '0';
    }
    set_replace_char(c);
    return result;
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
            g_token.type = PLUS_T;
            break;
        }
        case '-':
        {
            g_token.type = MINUS_T;
            break;
        }
        case '*':
        {
            g_token.type = MUL_T;
            break;
        }
        case '/':
        {
            g_token.type = DIV_T;
            break;
        }
        default:
        {
            // will replace with next_digit() here.
            if (isdigit(c))
            {
                g_token.type = INT_T;
                g_parse_digit = next_digit(c);
                break;
            }
            else 
            {
                // parse_panic("Unrecognize token");
            }
        }
            
    }
    return c;
}

void scan_file()
{
    while (scan())
    {
        printf("Line#%d: <token: %s> ", line_number, tokens[g_token.type]);
        if(g_token.type == INT_T)
        {
            printf("<value>: %d", g_parse_digit);
        }
        printf("\n");
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