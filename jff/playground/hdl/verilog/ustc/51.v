// https://verilogoj.ustc.edu.cn/oj/problem/99
module top_module(
    input 			clk		,  //4Hz
    input 			reset	,
	output	[7:0]	ss
); 
	// Write your code here
    reg cnt_low = 0;
    reg [4:0] cnt_high = 0;
    reg [3:0] low_ss, high_ss;
    assign ss = 8'hff;
    always @(posedge clk) begin
        if(cnt_high % 10) // 4 Hz, 4 time a loop
        if(reset) begin
            
        end
    end
	
endmodule
