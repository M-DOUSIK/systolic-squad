// tb_postproc — npu_postproc vs rq() vectors (gen_pp_vectors.py). Batches have static params; vectors inside a batch go
// back-to-back with random idle gaps; the pipe is drained between batches. Checks value, out_raw, exact latency (3),
// no invented/missing outputs.
`timescale 1ns/1ps
module tb_postproc #(
  parameter int N = 16, parameter int NB = 400, parameter int K = 25, parameter string PFX = "build/pp_N16"
);
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;

  logic in_valid = 0; logic [N*32-1:0] in_acc = 0;
  logic rq_en = 1, relu_en = 0;
  logic [N*16-1:0] rq_mult = 0; logic [N*5-1:0] rq_shift = 0; logic [N*32-1:0] bias = 0;
  logic signed [7:0] out_zp = 0;
  wire out_valid; wire [N*32-1:0] out_raw; wire [N*8-1:0] out_q;

  npu_postproc #(.N(N)) dut (.*);

  reg [31:0] acc_mem  [0:NB*K*N-1];
  reg [7:0]  exp_mem  [0:NB*K*N-1];
  reg [15:0] m_mem    [0:NB*N-1];
  reg [4:0]  s_mem    [0:NB*N-1];
  reg [31:0] bias_mem [0:NB*N-1];
  reg [7:0]  zp_mem   [0:NB-1];
  reg        relu_mem [0:NB-1];

  integer errors = 0, ocount = 0, icount = 0, cases = 0;
  reg [2:0] vp = 0;
  string f;

  initial begin
    f = {PFX, "_acc.hex"};  $readmemh(f, acc_mem);
    f = {PFX, "_exp.hex"};  $readmemh(f, exp_mem);
    f = {PFX, "_m.hex"};    $readmemh(f, m_mem);
    f = {PFX, "_s.hex"};    $readmemh(f, s_mem);
    f = {PFX, "_bias.hex"}; $readmemh(f, bias_mem);
    f = {PFX, "_zp.hex"};   $readmemh(f, zp_mem);
    f = {PFX, "_relu.hex"}; $readmemh(f, relu_mem);
  end

  // monitor: samples what the consumer sees at each posedge
  always @(posedge clk) begin
    if (!rst) begin
      if (out_valid !== vp[2]) begin
        errors = errors + 1;
        if (errors < 20) $display("LATENCY/VALID MISMATCH at t=%0t out_valid=%b expected %b", $time, out_valid, vp[2]);
      end
      if (out_valid === 1'b1) begin
        for (int c = 0; c < N; c++) begin
          cases = cases + 1;
          if (out_q[c*8 +: 8] !== exp_mem[ocount*N + c] || out_raw[c*32 +: 32] !== acc_mem[ocount*N + c]) begin
            errors = errors + 1;
            if (errors < 20) $display("MISMATCH vec %0d col %0d: q got %0d exp %0d | raw got %0h exp %0h",
              ocount, c, $signed(out_q[c*8 +: 8]), $signed(exp_mem[ocount*N + c]), out_raw[c*32 +: 32], acc_mem[ocount*N + c]);
          end
        end
        ocount = ocount + 1;
      end
    end
    vp <= {vp[1:0], in_valid};
  end

  initial begin
    repeat (4) @(negedge clk);
    rst = 0;
    for (int b = 0; b < NB; b++) begin
      @(negedge clk);
      out_zp  = zp_mem[b];
      relu_en = relu_mem[b];
      for (int c = 0; c < N; c++) begin
        rq_mult [c*16 +: 16] = m_mem[b*N + c];
        rq_shift[c*5  +: 5]  = s_mem[b*N + c];
        bias    [c*32 +: 32] = bias_mem[b*N + c];
      end
      for (int v = 0; v < K; v++) begin
        while ($urandom_range(99) < 25) begin in_valid = 0; @(negedge clk); end
        in_valid = 1;
        for (int c = 0; c < N; c++) in_acc[c*32 +: 32] = acc_mem[(b*K + v)*N + c];
        icount = icount + 1;
        @(negedge clk);
      end
      in_valid = 0; in_acc = {N{32'hDEADBEEF}};
      repeat (5) @(negedge clk);
      if (ocount != icount) begin errors = errors + 1; $display("COUNT MISMATCH batch %0d: in %0d out %0d", b, icount, ocount); end
    end
    $display("tb_postproc: N=%0d vectors=%0d cases=%0d errors=%0d", N, icount, cases, errors);
    if (errors == 0) $display("RESULT: PASS"); else $display("RESULT: FAIL");
    $finish;
  end
endmodule
