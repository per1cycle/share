#ifndef HELPER_H
#define HELPER_H

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