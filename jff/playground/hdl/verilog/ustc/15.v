// https://verilogoj.ustc.edu.cn/oj/problem/53
module top_module (
    input [4:0] a, b, c, d, e, f,
    output [7:0] w, x, y, z );
    wire [31:0] tmp;
    assign tmp[31:0] = {a, b, c, d, e, f, 2'b11};
    assign w[7:0] = tmp[31:24];
    assign x[7:0] = tmp[23:16];
    assign y[7:0] = tmp[15:8];
    assign z[7:0] = tmp[7:0];
endmodule
