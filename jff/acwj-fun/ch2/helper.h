#ifndef HELPER_H
#define HELPER_H

enum TOKEN_TYPE {
    PLUS_T = 0,
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


#endif // HELPER_H