// tb_npu_top — replays an APB command script from hw/tb/shell/gen_shell_tests.py against npu_top (contract v1.2) and checks
// every expected read. Also checks that the tag delay line is aligned with the array output on every clock.
//   iverilog -g2012 -P tb_npu_top.N=16 -P tb_npu_top.ACC_DEPTH=64 -P 'tb_npu_top.SCRIPT="build/shell_N16.cmd"' ...
`timescale 1ns/1ps
module tb_npu_top;
  parameter int    N         = 16;
  parameter int    ACC_DEPTH = 64;
  parameter int    XR_DEPTH  = 256;
  parameter string SCRIPT    = "build/shell.cmd";

  logic        clk = 0, presetn = 0;
  logic        psel = 0, penable = 0, pwrite = 0;
  logic [31:0] paddr = 0, pwdata = 0, prdata;
  logic        pready, pslverr;
  logic [7:0]  leds;
  always #5 clk = ~clk;

  npu_top #(.N(N), .ACC_DEPTH(ACC_DEPTH), .XR_DEPTH(XR_DEPTH)) dut (
    .PCLK(clk), .PRESETN(presetn), .PSEL(psel), .PENABLE(penable), .PWRITE(pwrite), .PADDR(paddr), .PWDATA(pwdata),
    .PRDATA(prdata), .PREADY(pready), .PSLVERR(pslverr), .leds(leds));

  int errors = 0, checks = 0, align_err = 0;
  longint cycles = 0;
  always @(posedge clk) begin
    cycles++;
    if (presetn && (dut.t_valid !== dut.sa_out_valid)) align_err++;
  end

  task automatic apb_wr(input logic [31:0] a, input logic [31:0] d);
    @(negedge clk); psel = 1; penable = 0; pwrite = 1; paddr = a; pwdata = d;
    @(negedge clk); penable = 1;
    @(negedge clk); psel = 0; penable = 0; pwrite = 0;
  endtask

  task automatic apb_rd(input logic [31:0] a, output logic [31:0] d);
    @(negedge clk); psel = 1; penable = 0; pwrite = 0; paddr = a;
    @(negedge clk); penable = 1;
    #1 d = prdata;
    @(negedge clk); psel = 0; penable = 0;
  endtask

  initial begin
    int fd, n, polls, done;
    reg [8*4-1:0] op;
    reg [8*120-1:0] txt;
    logic [31:0] a, v, m, d;
    fd = $fopen(SCRIPT, "r");
    if (fd == 0) begin $display("tb_npu_top: cannot open %s", SCRIPT); $finish; end
    repeat (5) @(negedge clk);
    presetn = 1;
    repeat (3) @(negedge clk);
    done = 0;
    while (!done) begin
      n = $fscanf(fd, "%s", op);
      if (n != 1 || op == "e") done = 1;
      else if (op == "c") begin n = $fscanf(fd, "%s", txt); $display("[N=%0d] -- %s", N, txt); end
      else begin
      n = $fscanf(fd, "%h %h %h", a, v, m);
      case (op)
        "w": apb_wr(a, v);
        "r": begin
          apb_rd(a, d);
          checks++;
          if ((d & m) !== (v & m)) begin
            errors++;
            if (errors <= 20) $display("MISMATCH addr %03h got %08h expected %08h mask %08h (t=%0t)", a, d, v, m, $time);
          end
        end
        "p", "l": begin
          polls = 0;
          do begin
            apb_rd(a, d);
            polls++;
          end while (((op == "p") ? ((d & m) !== v) : ((d & m) > v)) && polls < 100000);
          if (polls >= 100000) begin errors++; $display("TIMEOUT polling %03h for %08h mask %08h (last %08h)", a, v, m, d); $finish; end
        end
        default: begin $display("bad op %s", op); $finish; end
      endcase
      end
    end
    if (align_err != 0) begin errors++; $display("tag delay line misaligned on %0d clocks", align_err); end
    $display("tb_npu_top N=%0d ACC_DEPTH=%0d XR_DEPTH=%0d: %0d checks, %0d errors, %0d clocks", N, ACC_DEPTH, XR_DEPTH, checks, errors, cycles);
    $display("RESULT: %s", errors == 0 ? "PASS" : "FAIL");
    $finish;
  end
endmodule
