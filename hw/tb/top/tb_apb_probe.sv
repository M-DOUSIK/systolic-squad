`timescale 1ns/1ps
module tb_apb_probe;
    logic PCLK = 0, PRESETN = 0, PSEL = 0, PENABLE = 0, PWRITE = 0;
    logic [31:0] PADDR = 0, PWDATA = 0, PRDATA;  logic PREADY, PSLVERR;  logic [7:0] leds;
    integer errs = 0, i;  logic [31:0] d, v;
    always #10 PCLK = ~PCLK;
    apb_probe dut (.*);
    task apb_wr(input [31:0] a, input [31:0] dd);
        begin @(negedge PCLK); PSEL=1; PWRITE=1; PADDR=a; PWDATA=dd; PENABLE=0;
              @(negedge PCLK); PENABLE=1; @(negedge PCLK); PSEL=0; PENABLE=0; PWRITE=0; end
    endtask
    task apb_rd(input [31:0] a, output [31:0] dd);
        begin @(negedge PCLK); PSEL=1; PWRITE=0; PADDR=a; PENABLE=0;
              @(negedge PCLK); PENABLE=1; @(posedge PCLK); dd = PRDATA; @(negedge PCLK); PSEL=0; PENABLE=0; end
    endtask
    initial begin
        repeat (3) @(posedge PCLK); PRESETN = 1;
        apb_rd(32'h000, d); if (d !== 32'h53514431) begin errs++; $display("ID bad %h", d); end
        for (i = 0; i < 100; i++) begin v = $random; apb_wr(32'h044, v); apb_rd(32'h044, d); if (d !== v) errs++; end
        apb_wr(32'h048, 32'hA5); if (leds !== 8'hA5) errs++;
        apb_rd(32'h010, d); if (d !== 0) errs++;
        if (errs == 0) $display("RESULT: PASS apb_probe"); else $display("RESULT: FAIL apb_probe errs=%0d", errs);
        $finish;
    end
endmodule
