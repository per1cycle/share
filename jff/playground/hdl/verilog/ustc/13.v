// https://verilogoj.ustc.edu.cn/oj/problem/137
module top_module( 
    input [2:0] a,
    input [2:0] b,
    output [2:0] out_or_bitwise,
    output out_or_logical,
    output [5:0] out_not
);
// Write your code here
assign out_or_bitwise = a | b;
assign out_or_logical = a || b;
assign out_not[2:0] = ~a, out_not[5:3] = ~b;

endmodule
