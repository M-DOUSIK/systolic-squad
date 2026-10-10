// sa_pe — one processing element of the weight-stationary streaming array.
//   * activation hops right (a_out), partial sum hops down (ps_out), the new-weight flag rides with the activation
//   * active weight w_act + shadow weight sw_out; shadow chain runs down the column (sw_in -> sw_out when sw_shift)
//   * w_eff = flag_in ? shadow : active  -> the swap happens exactly when the flagged vector passes this PE
//   * ZERO_SKIP: when a_in == 0 the multiplier operands are held at their last used values (no toggling) and ps_in passes
//     through. w_act / flag keep updating, so results are bit-identical to ZERO_SKIP = 0.
//   * en = 0 freezes every register (SLEEP). Synchronous active-high reset.
module sa_pe #(
  parameter int ACC_W     = 32,
  parameter bit ZERO_SKIP = 1'b1
) (
  input  logic                     clk,
  input  logic                     rst,
  input  logic                     en,
  input  logic signed [7:0]        a_in,
  input  logic                     flag_in,
  input  logic signed [ACC_W-1:0]  ps_in,
  input  logic signed [7:0]        sw_in,
  input  logic                     sw_shift,
  output logic signed [7:0]        a_out,
  output logic                     flag_out,
  output logic signed [ACC_W-1:0]  ps_out,
  output logic signed [7:0]        sw_out
);
  logic signed [7:0]  w_act;
  logic signed [7:0]  w_eff;
  logic signed [7:0]  mul_a, mul_w;
  logic signed [15:0] prod;
  logic               skip;

  assign w_eff = flag_in ? sw_out : w_act;

  if (ZERO_SKIP) begin : g_zs
    logic signed [7:0] a_hold, w_hold;       // last operands that really went through the multiplier
    assign skip  = (a_in == 8'sd0);
    assign mul_a = skip ? a_hold : a_in;
    assign mul_w = skip ? w_hold : w_eff;
    always_ff @(posedge clk) begin
      if (rst) begin
        a_hold <= 8'sd0;
        w_hold <= 8'sd0;
      end else if (en && !skip) begin
        a_hold <= a_in;
        w_hold <= w_eff;
      end
    end
  end else begin : g_nozs
    assign skip  = 1'b0;
    assign mul_a = a_in;
    assign mul_w = w_eff;
  end

  assign prod = mul_a * mul_w;              // plain signed multiply + add: maps to one math block

  always_ff @(posedge clk) begin
    if (rst) begin
      a_out    <= 8'sd0;
      flag_out <= 1'b0;
      ps_out   <= '0;
      w_act    <= 8'sd0;
      sw_out   <= 8'sd0;
    end else if (en) begin
      a_out    <= a_in;
      flag_out <= flag_in;
      w_act    <= w_eff;
      ps_out   <= skip ? ps_in : ps_in + {{(ACC_W-16){prod[15]}}, prod};
      if (sw_shift) sw_out <= sw_in;
    end
  end
endmodule
