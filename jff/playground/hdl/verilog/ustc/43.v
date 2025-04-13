// https://verilogoj.ustc.edu.cn/oj/problem/141
module top_module(
  input clk,
  input resetn,
  input [1:0] byteena,
  input [15:0] d,
  output reg [15:0] q
);
    // Write your code here
    /*
     * 注意这里的 reset块判断完成以后只用begin end包起来 要不然只能说nanbeng。
     */
    always @(posedge clk) begin
        if(~resetn) begin
            q <= 16'h00;
        end
        else begin
            q[7:0] <= byteena[0] == 1? d[7:0] : q[7:0];
            q[15:8] <= byteena[1] == 1? d[15:8] : q[15:8];
        end

    end
endmodule