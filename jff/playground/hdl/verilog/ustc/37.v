// https://verilogoj.ustc.edu.cn/oj/problem/63
module top_module (
    input [7:0] in,
    output parity); 
    assign parity = ^in;
endmodule