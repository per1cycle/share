#include <iostream>
#include "common.hh"
template<int N>
struct fact
{
    static const long long value = N * fact<N - 1>::value;
};

template<>
struct fact<0>
{
    static const long long value = 1;
};


template<int N>
struct sum
{
    static const long long value = N + sum<N - 1>::value;
};

template<>
struct sum<0>
{
    static const long long value = 0;
};

int s(int n)
{
    if(n == 0) return 0;
    return n + s(n - 1);
}

int main()
{
    constexpr int n = 1023;
    Timer t;
    t.start();
    for(int i = 0; i < 10000; i ++)
        sum<n>::value;
        // s(n);
    t.stop();
    t.just_report_time();
    t.reset();

    return 0;
}