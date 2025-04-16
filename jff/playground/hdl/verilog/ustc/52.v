// https://verilogoj.ustc.edu.cn/oj/problem/107

module top_module(
    input clk,
    input areset,  //异步、高有效、复位值为0
    input load,
    input ena,
    input [3:0] data,
    output reg [3:0] q); 
//Write your code here
    always @(posedge clk or posedge areset) begin
        if(areset) begin
            q <= 0;
        end
        else begin
            
        end
    end
endmodule
