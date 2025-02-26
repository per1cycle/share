#ifndef HELPER_H
#define HELPER_H

struct token;

FILE* input_file;
FILE* output_file;
int line_number = 1;
char replace_char = 0;
struct token g_token;
int g_parse_digit = 0;
static const char output_file_name[] = "c.codegen";

enum TOKEN_TYPE {
    EOF_T = 0,
    ADD_T ,
    MINUS_T,
    MUL_T,
    DIV_T,
    INT_T,
    UNKNOWN_T,
};

struct token 
{
    enum TOKEN_TYPE type;
    int value;
};

const char* tokens_t[] = {
    "eof",
    "+",
    "-",
    "*",
    "/",
    "integer",
    "Unknown type",
};

const char* tokens_a[] = {
    "ast add",
    "ast minus",
    "ast mul",
    "ast div",
    "ast integer",
};
#endif // HELPER_H