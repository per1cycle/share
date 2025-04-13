// https://verilogoj.ustc.edu.cn/oj/problem/79
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
wire [15:0] lower, higher;

wire temp, unuse;

assign low_a = a[15:0], low_b = b[15:0];
assign high_a = a[31:16], high_b = b[31:16];

add16 l(low_a, low_b, 1'b0, lower, temp);
add16 h(high_a, high_b, temp, higher, unuse);

assign sum[15:0] = lower;
assign sum[31:16] = higher;
endmodule
