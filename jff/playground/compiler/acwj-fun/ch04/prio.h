#ifndef PRIO_H
#define PRIO_H

#include <stdio.h>
#include <stdlib.h>

static const int op_priority[] = {
    0, // EOF_T
    10, // ADD_T
    10, // MINUS_T
    20, // MUL_T
    20, // DIV_T
    0, // INT_T
};

int priority_of(int token_type)
{
    int priority = op_priority[token_type];
    if(priority == 0)
    {
        fprintf(stderr, "Unknown token type %d\n", token_type);
        exit(1);
    }
    return priority;
}

#endif // PRIO_H
