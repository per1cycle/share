// https://verilogoj.ustc.edu.cn/oj/problem/57
module add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout );
	assign {cout,sum} = a + b + cin;
endmodule
module top_module(
    input [31:0] a,
    input [31:0] b,
    input sub,
    output [31:0] sum
);
wire [31:0] true_b = b ^ {32{sub}};

wire [15:0] low_a, low_b;
wire [15:0] high_a, high_b;

wire [15:0] low_res, high_res;
wire cout, unuse;

assign low_a = a[15:0], low_b = true_b[15:0];
assign high_a = a[31:16], high_b = true_b[31:16];

add16 lower(low_a, low_b, sub, low_res, cout);
add16 higher(high_a, high_b, cout, high_res, unuse);

assign sum[15:0] = low_res;
assign sum[31:16] = high_res;
endmodule