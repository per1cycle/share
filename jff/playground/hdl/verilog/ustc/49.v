// https://verilogoj.ustc.edu.cn/oj/problem/78

module top_module (
    input 				clk		,	//时钟信号
    input 				reset	,   //同步高电平有效复位信号
    output reg 	[3:0] 	q			//计数结果
);
// 请用户在下方编辑代码
always @(posedge clk) begin
    if(reset) begin
        q <= 4'b0001;
    end
    else begin
        q <= q == 4'd10? 4'd1 : q + 4'b0001;
    end
end


//用户编辑到此为止 
endmodule
