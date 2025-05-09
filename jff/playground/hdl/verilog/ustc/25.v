// https://verilogoj.ustc.edu.cn/oj/problem/58

module add1 ( input a, input b, input cin,   output sum, output cout );
// Full adder module here
/*
a 0 0 1 1
b 0
c 0
o 0
*/
assign {cout, sum} = a + b + cin;
endmodule

// module add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout);
// wire c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15;

// add1 inst_0(.a(a[0]),.b(b[0]),.cin(cin),.sum(sum[0]),.cout(c1));
// add1 inst_1(.a(a[1]),.b(b[1]),.cin(c1),.sum(sum[1]),.cout(c2));
// add1 inst_2(.a(a[2]),.b(b[2]),.cin(c2),.sum(sum[2]),.cout(c3));
// add1 inst_3(.a(a[3]),.b(b[3]),.cin(c3),.sum(sum[3]),.cout(c4));
// add1 inst_4(.a(a[4]),.b(b[4]),.cin(c4),.sum(sum[4]),.cout(c5));
// add1 inst_5(.a(a[5]),.b(b[5]),.cin(c5),.sum(sum[5]),.cout(c6));
// add1 inst_6(.a(a[6]),.b(b[6]),.cin(c6),.sum(sum[6]),.cout(c7));
// add1 inst_7(.a(a[7]),.b(b[7]),.cin(c7),.sum(sum[7]),.cout(c8));
// add1 inst_8(.a(a[8]),.b(b[8]),.cin(c8),.sum(sum[8]),.cout(c9));
// add1 inst_9(.a(a[9]),.b(b[9]),.cin(c9),.sum(sum[9]),.cout(c10));
// add1 inst_10(.a(a[10]),.b(b[10]),.cin(c10),.sum(sum[10]),.cout(c11));
// add1 inst_11(.a(a[11]),.b(b[11]),.cin(c11),.sum(sum[11]),.cout(c12));
// add1 inst_12(.a(a[12]),.b(b[12]),.cin(c12),.sum(sum[12]),.cout(c13));
// add1 inst_13(.a(a[13]),.b(b[13]),.cin(c13),.sum(sum[13]),.cout(c14));
// add1 inst_14(.a(a[14]),.b(b[14]),.cin(c14),.sum(sum[14]),.cout(c15));
// add1 inst_15(.a(a[15]),.b(b[15]),.cin(c15),.sum(sum[15]),.cout(cout));
// endmodule

module top_module (
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum);
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

