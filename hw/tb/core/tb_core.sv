// tb_core — self-checking sweep testbench for sa_core (extends reference/tb/tb_sa.v).
// Streams M1 + M2 random vectors through two weight sets (second set shifted into the shadow regs while the first still
// streams, swapped by the flag riding the first vector of set 2). Adds over the reference test:
//   ZERO_PCT : percentage of activation elements forced to 0 (exercises ZERO_SKIP)
//   GAP_PCT  : percentage of clocks that are bubbles (in_valid = 0); weight shifts are scheduled by clock, not by vector
//   exact per-vector latency check (every out_valid exactly 2N-1 clocks after its input), no invented outputs
//   ZERO_SKIP: forwarded to the DUT, so the same file proves ZERO_SKIP = 0 and 1 give identical (correct) results
`timescale 1ns/1ps
module tb_core;
  parameter int N         = 4;
  parameter int ACC       = 32;
  parameter int M1        = 64;     // vectors on weight set 1 (>= 3N)
  parameter int M2        = 64;
  parameter bit ZERO_SKIP = 1'b1;
  parameter int ZERO_PCT  = 0;
  parameter int GAP_PCT   = 0;
  parameter int SEED      = 1;
  localparam int M = M1 + M2;

  logic clk = 0, rst = 1;
  always #5 clk = ~clk;

  logic              in_valid = 0, in_newW = 0, w_shift = 0;
  logic [N*8-1:0]    in_vec = 0, w_row = 0;
  logic              out_valid;
  logic [N*ACC-1:0]  out_vec;

  sa_core #(.N(N), .ACC_W(ACC), .PE_TYPE(0), .ZERO_SKIP(ZERO_SKIP)) dut (
    .clk(clk), .rst(rst), .in_valid(in_valid), .in_newW(in_newW), .in_vec(in_vec),
    .w_shift(w_shift), .w_row(w_row), .out_valid(out_valid), .out_vec(out_vec));

  integer W1 [N*N];
  integer W2 [N*N];
  integer V  [M*N];
  integer EXP[M*N];
  integer in_cyc [M];
  integer seed, cyc, errs, nout, sent, rel, i, j, k, acc, nzero;

  function automatic integer rnd(input integer mod);
    integer r;
    begin r = $random(seed); rnd = (r & 32'h7fffffff) % mod; end
  endfunction

  initial cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  initial begin
    seed = SEED; errs = 0; nout = 0; sent = 0; nzero = 0;
    for (i = 0; i < N*N; i++) begin W1[i] = rnd(199) - 99; W2[i] = rnd(199) - 99; end
    for (i = 0; i < M*N; i++) begin
      if (rnd(100) < ZERO_PCT) begin V[i] = 0; nzero++; end
      else V[i] = rnd(199) - 99;
    end
    for (k = 0; k < M; k++)
      for (j = 0; j < N; j++) begin
        acc = 0;
        for (i = 0; i < N; i++) acc += V[k*N+i] * ((k < M1) ? W1[i*N+j] : W2[i*N+j]);
        EXP[k*N+j] = acc;
      end

    repeat (3) @(posedge clk);
    #1 rst = 0;

    // weight set 1 into the shadow registers (row N-1 first)
    for (i = 0; i < N; i++) begin
      @(negedge clk);
      for (j = 0; j < N; j++) w_row[j*8 +: 8] = W1[(N-1-i)*N + j];
      w_shift = 1;
    end
    @(negedge clk); w_shift = 0;

    // stream; rel = clocks since the first vector was accepted. Set 2 is shifted during rel = 2N .. 3N-1.
    rel = 0;
    while (sent < M) begin
      @(negedge clk);
      if (rnd(100) < GAP_PCT && sent > 0) begin
        in_valid = 0; in_newW = 0; in_vec = '0;
      end else begin
        in_valid = 1;
        in_newW  = (sent == 0) || (sent == M1);
        for (j = 0; j < N; j++) in_vec[j*8 +: 8] = V[sent*N+j];
        in_cyc[sent] = cyc;
        sent++;
      end
      if (rel >= 2*N && rel < 3*N) begin
        for (j = 0; j < N; j++) w_row[j*8 +: 8] = W2[(N-1-(rel-2*N))*N + j];
        w_shift = 1;
      end else w_shift = 0;
      rel++;
    end
    @(negedge clk); in_valid = 0; in_newW = 0; w_shift = 0;
    repeat (4*N) @(posedge clk);

    $display("N=%0d ZERO_SKIP=%0d ZERO_PCT=%0d (%0d zero elems) GAP_PCT=%0d: %0d vectors, %0d results", N, ZERO_SKIP, ZERO_PCT, nzero, GAP_PCT, M, nout);
    if (errs == 0 && nout == M) $display("RESULT: PASS");
    else                        $display("RESULT: FAIL errs=%0d nout=%0d", errs, nout);
    $finish;
  end

  // monitor: in-order results, exact latency
  always @(posedge clk) begin
    if (!rst && out_valid) begin
      if (nout >= M) begin errs++; $display("EXTRA output (vector %0d)", nout); end
      else begin
        if (cyc - in_cyc[nout] != 2*N-1) begin
          errs++;
          if (errs < 6) $display("LATENCY vec %0d: %0d clocks, expected %0d", nout, cyc - in_cyc[nout], 2*N-1);
        end
        for (int c = 0; c < N; c++)
          if ($signed(out_vec[c*ACC +: ACC]) !== EXP[nout*N+c]) begin
            errs++;
            if (errs < 6) $display("MISMATCH vec %0d col %0d got %0d exp %0d", nout, c, $signed(out_vec[c*ACC +: ACC]), EXP[nout*N+c]);
          end
      end
      nout++;
    end
  end
endmodule
