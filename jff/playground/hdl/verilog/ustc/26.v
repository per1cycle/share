// https://verilogoj.ustc.edu.cn/oj/problem/91
module add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout );
	assign {cout,sum} = a + b + cin;
endmodule

module top_module(
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum
);
wire [15:0] low_a, low_b;
wire [15:0] high_a, high_b;
wire [15:0] lower, upper;
wire [15:0] higher1, higher2;

wire temp, unuse;

assign low_a = a[15:0], low_b = b[15:0];
assign high_a = a[31:16], high_b = b[31:16];

add16 l(low_a, low_b, 1'b0, lower, temp);
add16 h1(high_a, high_b, 1'b0, higher1, unuse);
add16 h2(high_a, high_b, 1'b1, higher2, unuse);

assign upper = (temp == 1'b0) ? higher1 : higher2;

assign sum[15:0] = lower;
assign sum[31:16] = upper;

endmodule