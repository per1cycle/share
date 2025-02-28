#include "lest.h"

int Fib(int n)
{
    if (n <= 1)
        return n;
    return Fib(n - 1) + Fib(n - 2);
}

int Factorial(int n)
{
    if (n <= 1)
        return 1;
    return n * Factorial(n - 1);
}

TEST(FibonacciTest, Positive)
{
    EXPECT(Fib(0), 0);
    EXPECT(Fib(1), 1);
    EXPECT(Fib(2), 1);
    EXPECT(Fib(3), 2);
    EXPECT(Fib(4), 3);
    EXPECT(Fib(5), 5);
    EXPECT(Fib(6), 8);
    EXPECT(Fib(7), 13);
    EXPECT(Fib(8), 21);
    EXPECT(Fib(9), 34);
}

TEST(FactorialTest, Positive)
{
    EXPECT(Factorial(0), 1);
    EXPECT(Factorial(1), 1);
    EXPECT(Factorial(2), 2);
    EXPECT(Factorial(3), 6);
    EXPECT(Factorial(4), 24);
    EXPECT(Factorial(5), 120);
    EXPECT(Factorial(6), 720);
    EXPECT(Factorial(7), 5040);
    EXPECT(Factorial(8), 40320);
    EXPECT(Factorial(9), 362880);
}