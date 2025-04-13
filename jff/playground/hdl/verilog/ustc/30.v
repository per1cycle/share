// https://verilogoj.ustc.edu.cn/oj/problem/67
module top_module(
    input a,
    input b,
    input sel_b1,
    input sel_b2,
    output wire out_assign,
    output reg out_always); 
// 请用户在下方编辑代码
assign out_assign = sel_b1? sel_b2? b: a : a;

always @(*) begin
    if(sel_b1)
        if(sel_b1)
            out_always = b;
        else 
            out_always = a;
    else 
        out_always = a;

end

//用户编辑到此为止
endmodule