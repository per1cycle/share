#ifndef AST_H
#define AST_H
#include "helper.h"
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

// ast operation
enum
{
    ADD_A,
    MINUS_A,
    MUL_A,
    DIV_A
};

struct ast_node
{
    int token_type;
    int value;
    struct ast_node *left_node;
    struct ast_node *right_node;
};

struct ast_node *make_node(token tk, struct ast_node *left_node, struct ast_node *right_node, int value)
{
    struct ast_node *node;
    node = (struct ast_node*)calloc(1, sizeof(ast_node));

    if(node == NULL)
    {
        fprintf(stderr, "[ERROR]: allocate ast_node error!");
        exit(1);
    }
    node->token_type = tk;
    node->left_node = left_node;
    node->right_node = right_node;
    node->value = value;
    return node;
}

int interprete_ast_tree(struct ast_node *node)
{
    int left_tree_value = 0, right_tree_value = 0;
    if(node->left_node)
    {
        left_tree_value = interprete_ast_tree(node->left_node);
    }

    if(node->right_node)
    {
        right_tree_value = interprete_ast_tree(node->right_node);
    }

    switch (node->token_type)
    {
        case PLUS_T:
            return (left_tree_value + right_tree_value);
            break;
        case MINUS_T:
            return (left_tree_value - right_tree_value);
            break;
        case MUL_T:
            return (left_tree_value * right_tree_value);
            break;
        case DIV_T:
            return (left_tree_value / right_tree_value);
            break;
        default:
            fprintf(stderr, "[ERROR]: ast tree parse error");
    }
    // should use
    return -1;

}

#endif // AST_H