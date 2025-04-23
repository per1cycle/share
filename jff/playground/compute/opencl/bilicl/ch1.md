# Bilicl

>  记录一下自己学习opencl的过程

OpenCL 是Khronos Group提出的异构计算框架，可以在绝大多数设备上运行，如
- n卡
- a卡
- cpu

不过有些很新的npu上面好像不行？

很久很久以前，那时候cpu的主频还处在一个比较低的阶段，那时候，人们认为单核的性能是可以无限提升的。
直到遇上了物理屏障，于是并行计算开始变得热门起来了。

在现在ai时代，大多数都是一下矩阵乘法的操作，而众所周知
在这个（2x3）和 （3x4） 的矩阵乘法中
```text
A = [
[1, 2, 3],
[2, 3, 4]
]

mul

B = [
[1, 0, 1, 0],
[0, 1, 0, 1],
[1, 2, 3, 4]
]
```

A的每一行乘 B的每一列是可以与别的操作无关的

在串行系统中 这个矩阵乘法很容易做

```c
// assuming a is N x M matrix
//          b is M x K matrix
//          result is N x K matrix
for(int i = 0; i < N; i ++) // for each row in a 
{
    for(int j = 0; j < K; j ++) // for each column in b
    {
        // for every 1 x M and M x 1 matmul
        for(int k = 0; k < M; k ++)
        {
            c[i][j] = 
                a[i][k] // i row element.
                +
                b[k][j] // j th column element.
                ;
        }
    }
}

```

但是在做 a第i行 和 b第j列的操作时候，其他操作被当前操作阻塞，但是其实是可以并行操作的。

- 在cpu中，计算操作都在 soc内部完成，而更广义的运算系统 如gpgpu 通用图形处理单元可以美滋滋把这个操作分发给硬件内的计算单元
- 在n卡中，这些计算单元叫Streaming Processor / cuda core
- 在a卡中，这些计算单元叫ZLUDA

整个gpu构成了一个simd单元

再看一个例子
```
foo = 1 + 2
bar = 5 + 5
fun = foo + bar
```

在这个例子中，foo bar的计算都是独立的，因此非常适合分发给计算单元。
而fun依赖于foo bar，因此需要设置一个拦路虎，需要集齐 foo bar两员大将方可干掉这个拦路虎，继续通行！

本系列参考《Opencl异构计算》一书， 2012版。
