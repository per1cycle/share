// https://verilogoj.ustc.edu.cn/oj/problem/96
module top_module(
    input [7:0] a,
    input [7:0] b,
    input [7:0] c,
    input [7:0] d,
    output [7:0] min
);
wire [7:0] min_ab = a > b? b : a;
wire [7:0] min_cd = c > d? d : c;

assign min = min_ab > min_cd? min_cd: min_ab;
endmodule