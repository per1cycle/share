// https://verilogoj.ustc.edu.cn/oj/problem/99
module top_module(
    input 			clk		,  //4Hz
    input 			reset	,
	output	[7:0]	ss
); 
	// Write your code here
    reg [4:0] cnt = 0;
    reg [3:0] low_ss, high_ss;
    assign ss = 8'hff;
    always @(posedge clk) begin
        cnt <= cnt + 1;
        if(reset) begin
            low_ss <= 0;
            high_ss <= 0;
        end
        else begin
            if(cnt % 4 == 0) begin
                cnt <= 0;
                low_ss <= low_ss + 1;
            end
            else if (low_ss % 10) begin
                high_ss <= high_ss + 1;
                low_ss <= 0;
            end
            else if ()
        end
    end
	
endmodule
