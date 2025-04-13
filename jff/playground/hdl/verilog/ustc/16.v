// https://verilogoj.ustc.edu.cn/oj/problem/89
module top_module (
    input [7:0] in,
    output [7:0] out
);
genvar i;
generate
for(i = 0; i < 8; i = i + 1) begin
    assign out[i] = in[8 - i - 1];
end
endgenerate

endmodule