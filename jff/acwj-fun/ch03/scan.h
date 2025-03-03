#ifndef SCAN_H
#define SCAN_H

#include <ctype.h>

#include "helper.h"
#include "ast.h"

extern struct token g_token;
extern int g_parse_digit;
extern int line_number;
extern char replace_char;
extern FILE* input_file;

int scan();
int next();
int next_digit(int start_digit);
void parse_panic(const char* panic_info);
void set_replace_char(int c);
void scan_file();
int skip();

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

void parse_panic(const char* panic_info)
{
    printf("[ERROR at line <#%d> ]: %s\n", line_number, panic_info);
    for(int i = 0; i < 1e8; i++);
}

void set_replace_char(int c)
{
    replace_char = c;
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

void scan_file()
{
    while (scan())
    {
        if (g_token.type == UNKNOWN_T)
        {
            continue;
        }
        printf("Line#%d: <token: %s> ", line_number, tokens_t[g_token.type]);
        if(g_token.type == INT_T)
        {
            printf("<value>: %d", g_parse_digit);
        }
        printf("\n");
    }
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
            g_token.type = EOF_T;
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
                return g_parse_digit;
            }
            else 
            {
                g_token.type = UNKNOWN_T;
                parse_panic("Parse error.");
                return 1;
            }
        }
            
    }
    return 1;
}


#endif // SCAN_H