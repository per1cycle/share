#ifndef CODEGEN_H
#define CODEGEN_H
#include "helper.h"
#include "ast.h"
extern FILE* output_file;
// TODO
/**
 * code generator structure.
 */
// typedef struct codegen
// {
//     int (*codegen_add)(int reg1, int reg2);
//     int (*codegen_minus)(int reg1, int reg2);
//     int (*codegen_mul)(int reg1, int reg2);
//     int (*codegen_div)(int reg1, int reg2);
//     int (*codegen_load)(int value);
//     int (*codegen_printint)(int reg);
//     int (*codegen_pre_asm)();
//     int (*codegen_post_asm)();
// } codegen;
enum REGISTERS 
{
    W8 = 0,
    W9,
    W10,
    W11,
    W12,
    W13,
    W14,
    W15,
    REGISTERS_NUM
};
// status of a register, 0 if free, 1 if used.
int regs[REGISTERS_NUM];
const char *registers[] = 
{
    "w8", "w9", "w10", "w11", "w12", "w13", "w14", "w15",
};

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

void free_all_registers()
{
    for(int i = 0; i < REGISTERS_NUM; i++)
        regs[i] = 0;
}

int allocate_register()
{
    for(int i = 0; i < REGISTERS_NUM; i++)
    {
        if(regs[i] == 0)
        {
            regs[i] = 1;
            return i;
        }
    }
    return -1;
}

int codegen_pre_asm()
{
    free_all_registers();
    fprintf(output_file, 
        "\t.global _main\n"
        "\t.p2align 2\n"
        "\n"
        "_print:\n"
        "\tsub\tsp, sp, #32\n"
        "\tstp\tx29, x30, [sp, #16]\n"
        "\tadd\tx29, sp, #16\n"
        "\tstur\tw0, [x29, #-4]\n"
        "\tldur\tw9, [x29, #-4]\n"
        "\tmov\tx8, x9\n"
        "\tmov\tx9, sp\n"
        "\tstr\tx8, [x9]\n"
        "\tadrp\tx0, l_.str@PAGE\n"
        "\tadd\tx0, x0, l_.str@PAGEOFF\n"
        "\tbl\t_printf\n"
        "\tldp\tx29, x30, [sp, #16]\n"
        "\tadd\tsp, sp, #32\n"
        "\tret\n"
        "\n"
        "l_.str:\n"
        "\t.asciz  \"%%d\"\n"
        "\n"
        "\t_main:\n"

    );

    return 1;
}

int codegen_printint(int result)
{
    fprintf(output_file, "mov w0, %d\n", result);
    return 1;
}

int codegen_post_asm()
{
    fprintf(output_file, 
        "\n"
        "l_.str:\n"
        "\t.string \"%%d\\n\"\n"
        "\n"
    );
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
