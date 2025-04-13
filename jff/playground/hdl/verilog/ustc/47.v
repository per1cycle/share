// https://verilogoj.ustc.edu.cn/oj/problem/64
module top_module (
    input clk,
    input in,
    output reg out
); 
reg tmp;
reg cnt;
initial begin
    tmp = 0;
    cnt = 0;
end

always @(posedge clk) begin
    if(cnt < 1) begin
        cnt <= cnt + 1;
    end
    else begin
        if(tmp ^ in) begin
            out <= 1;
        end
        else begin
            out <= 0;
        end
        tmp <= in;
    end

end

endmodule
