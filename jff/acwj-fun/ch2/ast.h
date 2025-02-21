#ifndef AST_H
#define AST_H
#include "helper.h"

struct ast_node
{
    struct token token_type;
    int value;
    struct ast_node *left_node;
    struct ast_node *right_node;
};



#endif // AST_H