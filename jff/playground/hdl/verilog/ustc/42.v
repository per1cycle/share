// https://verilogoj.ustc.edu.cn/oj/problem/106
module top_module (
    input clk,
    input areset,   // active high asynchronous reset
    input [7:0] d,
    output reg [7:0] q
);
// Write your code here
always @(posedge clk or posedge areset) begin
    if(areset)
        q <= 8'd0;
    else 
        q <= d;

end
endmodule
