// https://verilogoj.ustc.edu.cn/oj/problem/60
module top_module( 
    input a, 
    input b, 
    output out );
// 请用户在下方编辑代码
/**
a 0 0 1 1
b 0 1 0 1
o 1 0 0 1
*/
assign out = ~(a ^ b);
//用户编辑到此为止
endmodule