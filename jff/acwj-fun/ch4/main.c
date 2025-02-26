#include <stdio.h>
#include <stdlib.h>

#include "ast.h"
#include "scan.h"
#include "prio.h"
#include "codegen.h"

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

struct ast_node* build_ast(int pervious_token_priority)
{
    struct ast_node *node, *left, *right;
    int node_type, token_type;

    left = first();
    token_type = g_token.type;

    if(g_token.type == EOF_T)
        return (left);

    while(priority_of(token_type) > pervious_token_priority)
    {

        right = build_ast(op_priority[token_type]);
        left = make_node(token_to_ast_op(token_type), left, right, 0);
        token_type = g_token.type;

        if(token_type == EOF_T)
            return left;
        
    }

    return (left);
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

    struct ast_node *node = build_ast(0);
    output_file = fopen(output_file_name, "w");

    if(output_file == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        exit(1);
    }
    
    gen_asm(node);

    return 0;
}