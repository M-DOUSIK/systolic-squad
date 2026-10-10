// tb_pe — self-checking unit test of sa_pe (ZERO_SKIP = 0 and 1 side by side, both checked against one integer model).
// Phases: (1) exhaustive 256 x 256 activation x weight products through the shadow -> active swap,
//         (2) random stress: random a (30% zeros), w, ps, flag, sw_shift, en and occasional reset.
// Run: iverilog -g2012 -o build/tb_pe hw/tb/core/tb_pe.sv hw/rtl/core/sa_pe.sv && vvp -n build/tb_pe
`timescale 1ns/1ps
module tb_pe;
  parameter int ACC_W   = 32;
  parameter int RANDOM_CYCLES = 200000;

  logic clk = 0, rst = 1, en = 1;
  always #5 clk = ~clk;

  logic signed [7:0]       a_in = 0, sw_in = 0;
  logic                    flag_in = 0, sw_shift = 0;
  logic signed [ACC_W-1:0] ps_in = 0;

  logic signed [7:0]       a_out   [2];
  logic                    flag_out[2];
  logic signed [ACC_W-1:0] ps_out  [2];
  logic signed [7:0]       sw_out  [2];

  for (genvar z = 0; z < 2; z++) begin : g_dut
    sa_pe #(.ACC_W(ACC_W), .ZERO_SKIP(z)) dut (
      .clk(clk), .rst(rst), .en(en),
      .a_in(a_in), .flag_in(flag_in), .ps_in(ps_in), .sw_in(sw_in), .sw_shift(sw_shift),
      .a_out(a_out[z]), .flag_out(flag_out[z]), .ps_out(ps_out[z]), .sw_out(sw_out[z]));
  end

  // ---- integer model, advanced at every posedge from the inputs that were set at the previous negedge ----
  logic signed [7:0]       m_wact = 0, m_sw = 0, m_weff;
  logic signed [7:0]       e_a = 0;
  logic                    e_flag = 0;
  logic signed [ACC_W-1:0] e_ps = 0;
  always @(posedge clk) begin
    if (rst) begin
      m_wact = 0; m_sw = 0; e_a = 0; e_flag = 0; e_ps = 0;
    end else if (en) begin
      m_weff = flag_in ? m_sw : m_wact;
      e_ps   = ps_in + a_in * m_weff;
      e_a    = a_in;
      e_flag = flag_in;
      m_wact = m_weff;
      if (sw_shift) m_sw = sw_in;
    end
  end

  integer errs = 0, checks = 0, n_zero = 0, n_hold = 0;

  // compare at negedge (outputs settled), then drive new inputs
  task automatic check;
    for (int z = 0; z < 2; z++) begin
      checks++;
      if (a_out[z] !== e_a || flag_out[z] !== e_flag || ps_out[z] !== e_ps || sw_out[z] !== m_sw) begin
        errs++;
        if (errs < 8)
          $display("MISMATCH ZS=%0d t=%0t a_out=%0d/%0d flag=%b/%b ps=%0d/%0d sw=%0d/%0d", z, $time,
                   a_out[z], e_a, flag_out[z], e_flag, ps_out[z], e_ps, sw_out[z], m_sw);
      end
    end
    if (ps_out[0] !== ps_out[1]) begin errs++; $display("MISMATCH ZS=0 vs ZS=1 differ at %0t", $time); end
  endtask

  task automatic step(input logic signed [7:0] a, input logic f, input logic signed [ACC_W-1:0] ps,
                      input logic signed [7:0] sw, input logic sh, input logic e, input logic r);
    @(negedge clk);
    check();
    a_in = a; flag_in = f; ps_in = ps; sw_in = sw; sw_shift = sh; en = e; rst = r;
  endtask

  integer seed = 1;
  integer w, a, i;
  logic signed [ACC_W-1:0] rps;

  initial begin
    repeat (3) @(negedge clk);
    step(0, 0, 0, 0, 0, 1, 0);

    // phase 1: exhaustive products. For every weight: shift it into the shadow reg, swap it in with a flagged
    // vector (a = -128), then run all 256 activations through the active weight with a random ps_in.
    for (w = -128; w < 128; w++) begin
      step(0, 0, 0, w, 1, 1, 0);
      step(-128, 1, $random(seed), 0, 0, 1, 0);
      for (a = -127; a < 128; a++) begin
        rps = $random(seed);
        step(a, 0, rps, 0, 0, 1, 0);
      end
    end

    // phase 2: random stress
    for (i = 0; i < RANDOM_CYCLES; i++) begin
      rps = $random(seed);
      if (($random(seed) & 32'h7fffffff) % 100 < 30) begin a = 0; n_zero++; end
      else a = $random(seed);
      step(a, (($random(seed) & 32'h7fffffff) % 100) < 40, rps, $random(seed),
           (($random(seed) & 32'h7fffffff) % 100) < 30,
           (($random(seed) & 32'h7fffffff) % 100) < 85,
           (($random(seed) & 32'h7fffffff) % 100) < 1);
    end
    step(0, 0, 0, 0, 0, 1, 0);
    step(0, 0, 0, 0, 0, 1, 0);

    $display("tb_pe: %0d checks, %0d zero activations driven in the random phase", checks, n_zero);
    if (errs == 0) $display("RESULT: PASS");
    else           $display("RESULT: FAIL errs=%0d", errs);
    $finish;
  end
endmodule
