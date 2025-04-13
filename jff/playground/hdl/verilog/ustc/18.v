// https://verilogoj.ustc.edu.cn/oj/problem/61
module top_module (
    input a, b, c, d, e,
    output [24:0] out );
    wire [24:0] foo = {{5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}}};
    wire [24:0] bar = {5{a, b, c, d, e}};
    assign out = foo ~^ bar;
endmodule