// https://verilogoj.ustc.edu.cn/oj/problem/88
module top_module (
    input clk,
    input reset,      // 异步复位，高电平有效，复位值为0
    output reg [3:0] q);

always @(posedge clk or posedge reset) begin
    if(reset) begin
        q <= 4'b0000;
    end
    else begin
        q <= (q == 4'b1111) ? 4'b0000 : q + 4'b0001;
    end
end
endmodule
