// https://verilogoj.ustc.edu.cn/oj/problem/97
module top_module (
    input clk,
    input in,
    output reg out
); 
reg tmp;
initial begin
    tmp = 0;
    out = 0;
end

always @(posedge clk) begin
    if(tmp != in && tmp == 0) begin
       tmp <= in; 
       out <= 1'b1;
    end
    else if(tmp != in && tmp == 1) begin
        tmp <= 0;
    end
    else begin
        out <= 1'b0;
    end

end

endmodule
