#ifndef HELPER_H
#define HELPER_H

enum TOKEN_TYPE {
    EOF_T = 0,
    PLUS_T ,
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

const char* tokens[] = {
    "eof",
    "+",
    "-",
    "*",
    "/",
    "integer",
    "Unknown type",
};

#endif // HELPER_H