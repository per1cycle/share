// https://verilogoj.ustc.edu.cn/oj/problem/88
module top_module (
    input clk,
    input reset,      // 异步复位，高电平有效，复位值为0
    output reg [3:0] q);
always @(posedge clk or posedge reset) begin
    
end
endmodule
