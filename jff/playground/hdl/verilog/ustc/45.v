// https://verilogoj.ustc.edu.cn/oj/problem/103
// module foo(
//     input clk,
//     input x,
//     output reg q,
//     output reg nq
// );
// always @(posedge clk) begin
//     q <= x;
//     nq <= ~x;
// end
// endmodule

module top_module (
    input clk,
    input x,
    output z
); 
reg q1;
reg q2;
reg q3;

initial begin
    q1 = 0;
    q2 = 0;
    q3 = 0;
end

always @(posedge clk) begin
    q1 <= x ^ q1;
    // IMPORTANT: always wrap the ~ with the variable.
    q2 <= (~q2) & x;
    q3 <= (~q3) | x;
end

assign z = ~(q1 | q2 | q3);

endmodule
