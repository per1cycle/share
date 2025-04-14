// https://verilogoj.ustc.edu.cn/oj/problem/70

module top_module(
    input clk,
    input reset,
    input en,
  	output reg [3:0]q);


always @(posedge clk) begin
	if(~en) begin
		q <= q;
	end
	else begin
		if(reset) begin
			q <= 4'd5;
		end
		else begin
			q <= (q == 4'd5)? 4'd15 : (q - 4'd1);
		end
	end
end

endmodule
