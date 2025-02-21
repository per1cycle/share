#ifndef AST_H
#define AST_H
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

// ast operation
enum
{
    ADD_A,
    MINUS_A,
    MUL_A,
    DIV_A,
    INT_A,
};

struct ast_node
{
    int token_type;
    int value;
    struct ast_node *left_node;
    struct ast_node *right_node;
};

struct ast_node *make_node(int token, struct ast_node *left_node, struct ast_node *right_node, int value)
{
    struct ast_node *node;
    node = (struct ast_node*)calloc(1, sizeof(ast_node));

    if(node == NULL)
    {
        fprintf(stderr, "[ERROR]: allocate ast_node error!");
        exit(1);
    }
    node->token_type = token;
    node->left_node = left_node;
    node->right_node = right_node;
    node->value = value;
    return node;
}

/**
 * left node will only be digit type.
 */
struct ast_node *make_leaf_node(int value)
{
    return make_node(INT_A, NULL, NULL, value);
}

struct ast_node *make_unary_node(int token, struct ast_node *left_children, int value)
{
    return make_node(token, left_children, NULL, value);
}

int token_to_ast_op(int token)
{
    switch (token)
    {
        case ADD_T:
            return (ADD_A);
        case MINUS_T:
            return (MINUS_A);
        case MUL_T:
            return (MUL_A);
        case DIV_T:
            return (DIV_A);
        case INT_T:
            return (INT_A);
        default:
            fprintf(stderr, "Invalid token");
    }
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
        case ADD_A:
            return (left_tree_value + right_tree_value);
            break;
        case MINUS_A:
            return (left_tree_value - right_tree_value);
            break;
        case MUL_A:
            return (left_tree_value * right_tree_value);
            break;
        case DIV_A:
            return (left_tree_value / right_tree_value);
            break;
        default:
            fprintf(stderr, "[ERROR]: ast tree parse error");
    }
    // shouldn't be used
    return -1;
}

#endif // AST_H