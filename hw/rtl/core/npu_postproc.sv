// npu_postproc — per-column bias + requantisation, bit-exact with rq() in
// reference/qmath_ref.py. Fixed latency PP_LAT = 3 clocks (in_valid in cycle c -> out_valid in cycle c+3), 1 vector/clock.
//   stage 1: sum  = acc + bias[c]                       (33-bit signed)
//   stage 2: prod = sum * {1'b0, M[c]}                  (49-bit signed, M unsigned 1..32767)
//   stage 3: v = (prod + (1<<(S-1))) >>> S  (S>0), + out_zp, saturate to [lo,127], lo = relu ? out_zp : -128
// out_raw = in_acc delayed 3 clocks (no bias). rq_en is a static mode bit for the shell: out_q is always computed,
// the shell picks out_raw or out_q. M, S, bias, out_zp, relu_en are sampled by the stage that uses them, so they must
// be static while vectors are in flight (contract: "static while busy").
module npu_postproc #(
  parameter int N = 16
) (
  input  logic                 clk,
  input  logic                 rst,
  input  logic                 in_valid,
  input  logic [N*32-1:0]      in_acc,
  input  logic                 rq_en,
  input  logic                 relu_en,
  input  logic [N*16-1:0]      rq_mult,
  input  logic [N*5-1:0]       rq_shift,
  input  logic [N*32-1:0]      bias,
  input  logic signed [7:0]    out_zp,
  output logic                 out_valid,
  output logic [N*32-1:0]      out_raw,
  output logic [N*8-1:0]       out_q
);
  logic v1, v2;
  logic [N*32-1:0] raw1, raw2;

  always_ff @(posedge clk) begin
    if (rst) begin
      v1 <= 1'b0; v2 <= 1'b0; out_valid <= 1'b0;
    end else begin
      v1 <= in_valid; v2 <= v1; out_valid <= v2;
    end
    raw1    <= in_acc;
    raw2    <= raw1;
    out_raw <= raw2;
  end

  for (genvar c = 0; c < N; c++) begin : g_col
    logic signed [32:0] sum1;
    logic signed [48:0] prod2;
    logic signed [48:0] rnd, shifted, withzp;
    logic        [4:0]  s_c;
    logic signed [48:0] zp_x, lo_b;

    always_ff @(posedge clk) begin
      sum1  <= $signed(in_acc[c*32 +: 32]) + $signed(bias[c*32 +: 32]);
      prod2 <= $signed({{16{sum1[32]}}, sum1}) * $signed({33'd0, rq_mult[c*16 +: 16]});
    end

    assign s_c     = rq_shift[c*5 +: 5];
    assign zp_x    = $signed({{41{out_zp[7]}}, out_zp});
    assign rnd     = (s_c == 5'd0) ? prod2 : (prod2 + (49'sd1 <<< (s_c - 5'd1)));
    assign shifted = rnd >>> s_c;
    assign withzp  = shifted + zp_x;
    assign lo_b    = relu_en ? zp_x : -49'sd128;

    always_ff @(posedge clk) begin
      if (withzp > 49'sd127)    out_q[c*8 +: 8] <= 8'd127;
      else if (withzp < lo_b)   out_q[c*8 +: 8] <= lo_b[7:0];
      else                      out_q[c*8 +: 8] <= withzp[7:0];
    end
  end

  logic unused_rq_en;
  assign unused_rq_en = rq_en;
endmodule
