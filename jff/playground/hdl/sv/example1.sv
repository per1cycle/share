// https://systemverilog.dev/1.html
module example1;

typedef enum logic[1:0]
{
    foo1,
    foo2,
    foo3,
    foo4
} stat_t;

enum logic[1:0] {ST0, ST1, ST2, ST3} st;

initial begin
    $display("hello world");
end

endmodule