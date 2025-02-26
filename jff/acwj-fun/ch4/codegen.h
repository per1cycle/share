#ifndef CODEGEN_H
#define CODEGEN_H
#include "helper.h"
#include "ast.h"
extern FILE* output_file;

int codegen_add(int reg1, int reg2);
int codegen_minus(int reg1, int reg2);
int codegen_mul(int reg1, int reg2);
int codegen_div(int reg1, int reg2);
int codegen_load(int value);

int codegen_load(int value)
{
    fprintf(output_file, "load rx, xxx\n");
    return 1;
}

int codegen_add(int reg1, int reg2)
{
    fprintf(output_file, "add xx, xx\n");
    return reg1;
}

int codegen_minus(int reg1, int reg2)
{
    fprintf(output_file, "subs xx, xx\n");
    return reg1;
}

int codegen_mul(int reg1, int reg2)
{
    fprintf(output_file, "mul xx, xx\n");
    return reg1;
}

int codegen_div(int reg1, int reg2)
{
    fprintf(output_file, "div x, x\n");
    return reg1;
}

int codegen_pre_asm()
{
    fprintf(output_file, "start\n");
    return 1;
}

int codegen_printint(int reg)
{
    fprintf(output_file, "printint rx\n");
    return 1;
}

int codegen_post_asm()
{
    fprintf(output_file, "end\n");
    return 1;
}

int gen_ast(struct ast_node* node)
{
    int left_result = 0, right_result = 0;
    if(node->left_node)
        left_result = gen_ast(node->left_node);
    if(node->right_node)
        right_result = gen_ast(node->right_node);
    
    switch(node->token_type)
    {
        case ADD_A:
            return codegen_add(left_result, right_result);
        case MINUS_A:
            return codegen_minus(left_result, right_result);
        case MUL_A:
            return codegen_mul(left_result, right_result);
        case DIV_A:
            return codegen_div(left_result, right_result);
        case INT_A:
            return codegen_load(node->value);
    }
    return 1;
}

int gen_asm(struct ast_node* node)
{
    int ret = 0;
    codegen_pre_asm();
    ret = gen_ast(node);
    codegen_printint(ret);
    codegen_post_asm();
    return ret;
}

#endif // CODEGEN_H
