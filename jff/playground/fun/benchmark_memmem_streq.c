#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int loop_time;
const int N = 100010;
typedef struct str_group
{
    char *str1; // \0 in the last
    char *str2;
} str_group_t;

str_group_t *test_group;

void usage()
{
    printf("Usage: ./a.out <loop time>.");
}

void genenrate_test_compatible_string()
{
    const char* dict = "abcdefghijklmnopqrstuvwxyz";
    int dict_len = strlen(dict);
    srand(time(NULL));
    test_group = (str_group_t*) malloc(sizeof(str_group_t) * loop_time);

    for(int i = 0; i < loop_time; i ++)
    {
        // comp name len = 10, subsystem name len = chip-subsys, chip len = 4, subsys name = 6.
        // so str = 10 + 1 + 4 + 1 + 6 = 22.
        test_group[i].str1 = (char*) malloc(sizeof(char) * 23);
        test_group[i].str2 = (char*) malloc(sizeof(char) * 23);
        int is_eq = rand() % 2;

        for(int iter = 0; iter < 23 ; iter ++)
        {
            if(iter == 10)
            {
                test_group[i].str1[iter] = ',';
                test_group[i].str2[iter] = ',';
            }
            else if (iter == 16) 
            {
                test_group[i].str1[iter] = '-';
                test_group[i].str2[iter] = '-';
            }
            else if (iter == 22)
            {
                test_group[i].str1[iter] = 0;
                test_group[i].str2[iter] = 0;
            }
            else
            {
                if(!is_eq)
                {
                    test_group[i].str1[iter] = dict[rand() % dict_len];
                    test_group[i].str2[iter] = dict[rand() % dict_len];
                }
                else
                {
                    test_group[i].str1[iter] = dict[rand() % dict_len];
                    test_group[i].str2[iter] = test_group[i].str1[iter];
                }
            }
        }
    }
}

void print_first_test_string(int first)
{
    if(first > loop_time)
    {
        fprintf(stderr, "Number should be less than the total test number.\n");
        exit(1);
    }

    for(int i = 0; i < first; i ++)
    {
        printf("%d test group in test_group: <%s>, <%s>.\n", i, test_group[i].str1, test_group[i].str2);
    }
}
void benchmark_strceq()
{
    printf("Start benchmark streq with loop: %d\n", loop_time);
    int eq_num = 0;

    for(int i = 0; i < loop_time; i ++)
    {
        if(strcmp(test_group[i].str1, test_group[i].str2) == 0)
        {
            eq_num ++;
        }
    }
    
    printf("Eq number is: %d\n", eq_num);
}

void benchmark_memmem()
{
    printf("Start benchmark memmem with loop: %d\n", loop_time);
    int eq_num = 0;

    for(int i = 0; i < loop_time; i ++)
    {
        if(memcmp((const void*)test_group[i].str1, (const void*)test_group[i].str2, strlen(test_group[i].str1)) == 0)
        {
            eq_num ++;
        }
    }
    
    printf("Eq number is: %d\n", eq_num);
}

void measure_time(void (*func)(void))
{
    clock_t start = clock();
    (*func)();
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;

    printf("Function costs: %.3f seconds\n", seconds);
}
int main(int argc, char ** argv)
{
    if(argc < 2)
    {
        usage();
        exit(1);
    }
    loop_time = atoi(argv[1]);
    printf("Loop time: %d\n", loop_time);
    
    measure_time(genenrate_test_compatible_string);
    // print_first_test_string(loop_time / 2);

    measure_time(benchmark_memmem);
    measure_time(benchmark_strceq);
    return 0;

}