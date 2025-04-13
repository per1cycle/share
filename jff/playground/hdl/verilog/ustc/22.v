// https://verilogoj.ustc.edu.cn/oj/problem/83
module my_dff(input clk,input d,output reg q);
	always@(posedge clk)
    	q <= d;
endmodule

module top_module ( input clk, input d, output q );
// Write your code here
wire tmp1, tmp2;
my_dff d1(clk, d, tmp1);
my_dff d2(clk, tmp1, tmp2);
my_dff d3(clk, tmp2, q);

endmodule
