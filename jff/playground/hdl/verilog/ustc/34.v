// https://verilogoj.ustc.edu.cn/oj/problem/108
module top_module (
    input [7:0] in,
    output reg [2:0] pos
);
always @(*) begin
    casez (in)
        8'bzzzzzzz1: pos = 3'd0;  // Position 0
        8'bzzzzzz10: pos = 3'd1;  // Position 1
        8'bzzzzz100: pos = 3'd2;  // Position 2
        8'bzzzz1000: pos = 3'd3;  // Position 3
        8'bzzz10000: pos = 3'd4;  // Position 4
        8'bzz100000: pos = 3'd5;  // Position 5
        8'bz1000000: pos = 3'd6;  // Position 6
        8'b10000000: pos = 3'd7;  // Position 7
        default: pos = 3'd0;      // Undefined position if no '1' is found
    endcase
end
endmodule