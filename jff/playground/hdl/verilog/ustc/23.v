// https://verilogoj.ustc.edu.cn/oj/problem/138
module my_dff8(
  input clk,
  input [7:0] d,
  output reg [7:0] q
);
	always@(posedge clk)
    	q <= d;
endmodule


module top_module(
  input clk,
  input [7:0] d,
  input [1:0] sel,
  output reg [7:0] q
);
// Write your code here
wire [7:0] temp1, temp2, temp3;

my_dff8 foo(clk, d, temp1);
my_dff8 foo1(clk, temp1, temp2);
my_dff8 foo2(clk, temp2, temp3);

always @(*) begin
    case (sel)
        2'b00: q = d; 
        2'b01: q = temp1;
        2'b10: q = temp2;
        2'b11: q = temp3;
    endcase
end

endmodule