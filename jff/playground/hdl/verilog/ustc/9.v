// https://verilogoj.ustc.edu.cn/oj/problem/65
module top_module(
    input a,
    input b,
    input c,
    input d,
    output out,
    output out_n   
); 
// 请用户在下方编辑代码
wire ab_and, cd_and;

assign out_n = ~(ab_and | cd_and);
assign out = ab_and | cd_and;
assign ab_and = a & b;
assign cd_and = c & d;
//用户编辑到此为止
endmodule
