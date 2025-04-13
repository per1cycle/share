// https://verilogoj.ustc.edu.cn/oj/problem/95
module top_module (
    input		cpu_overheated		,
    output	reg	shut_off_computer	,
    input      	arrived				,
    input      	gas_tank_empty		,
    output 	reg keep_driving
);
  	// Edit the code below
	always @(*) begin
    	if (cpu_overheated)
        	shut_off_computer = 1'b1;
        else
            shut_off_computer = 1'b0;
    end

    always @(*) begin
    	if (~arrived)
        	keep_driving = ~gas_tank_empty;
        else
            keep_driving = 1'b0;
    end
endmodule
