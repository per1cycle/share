#include <stdio.h>
#include <stdlib.h>

#include "ast.h"
#include "scan.h"

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

struct ast_node *first()
{
    int c = scan();

    struct ast_node *node;

    switch (g_token.type)
    {
    case INT_T:
        node = make_leaf_node(c);
        scan();
        return (node);
        break;
    
    default:
        exit(1);
        break;
    }
    return NULL;
}

struct ast_node* build_ast()
{
    struct ast_node *node, *left, *right;
    int node_type;

    left = first();

    if(g_token.type == EOF_T)
    {
        return (left);
    }

    node_type = token_to_ast_op(g_token.type);
    
    right = build_ast();
    
    node = make_node(node_type, left, right, 0);
    return (node);
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

    struct ast_node *node = build_ast();

    printf("%d\n", interprete_ast_tree(node));

    return 0;
}