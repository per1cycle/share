// https://verilogoj.ustc.edu.cn/oj/problem/68
module top_module(
    input clk,
    input a,
    input b,
    output wire out_assign,
    output reg out_always_comb,
    output reg out_always_ff   
);
// 请用户在下方编辑代码
assign out_assign = a ^ b;
always @(*) begin
   out_always_comb = a ^ b;
end

always @(posedge clk) begin
    out_always_ff <= a ^ b;
end
//用户编辑到此为止
endmodule