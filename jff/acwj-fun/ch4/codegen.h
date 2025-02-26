#ifndef CODEGEN_H
#define CODEGEN_H
#include "helper.h"
#include "ast.h"
extern FILE* output_file;

int codegen_add(int reg1, int reg2);
int codegen_minus(int reg1, int reg2);
int codegen_mul(int reg1, int reg2);
int codegen_div(int reg1, int reg2);



int gen_asm(struct ast_node* node)
{
    int left_result = 0, right_result = 0;
    if(node->left_node)
        left_result = gen_asm(node->left_node);
    if(node->right_node)
        right_result = gen_asm(node->right_node);
    
    switch(node->token_type)
    {
        case ADD_A:
            codegen_add(left_result, right_result);
            break;
        case MINUS_A:
            codegen_minus(left_result, right_result);
            break;
        case MUL_A:
            codegen_mul(left_result, right_result);
            break;
        case DIV_A:
            codegen_div(left_result, right_result);
            break;
    }
    return 1;
}

#endif // CODEGEN_H
