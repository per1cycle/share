#include <ctype.h>
#include <stdio.h>

#include "helper.h"

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
    printf("[ERROR at line <#%d> ]: %s\n", line_number, panic_info);
    for(int i = 0; i < 1e8; i++);
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
            g_token.type = ADD_T;
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
                g_token.type = UNKNOWN_T;
                parse_panic("Parse error.");
                return 1;
            }
        }
            
    }
    return c;
}

void scan_file()
{
    while (scan())
    {
        if (g_token.type == UNKNOWN_T)
        {
            continue;
        }
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