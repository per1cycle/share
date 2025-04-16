// https://verilogoj.ustc.edu.cn/oj/problem/99
module top_module(
    input 			clk		,  //4Hz
    input 			reset	,
	output	reg [7:0]	ss
); 
	// Write your code here
    reg [3:0] cnt = 0;
    reg [3:0] low_ss = 4'hf, high_ss = 4'hf;
    
    always @(posedge clk) begin
        cnt <= cnt + 1;
        if(reset) begin
            low_ss <= 4'b0000;
            high_ss <= 4'b0000;
            ss <= 8'h00;
            cnt = 0;
        end
        else begin
            if(cnt % 4 == 0) begin
                cnt = 0;
                low_ss <= low_ss + 1;
            end
            else begin
                if(low_ss == 10) begin
                    low_ss <= 0;
                    high_ss <= high_ss + 1;
                end
                else if (high_ss == 6) begin
                    high_ss <= 0;
                end
            end
            ss <= {high_ss, low_ss};
        end
    end
	
endmodule
