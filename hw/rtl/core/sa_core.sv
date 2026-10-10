// sa_core — N x N weight-stationary streaming systolic array.
//   y[c] = sum_r x[r] * W[r][c], one vector per clock, latency 2N-1 clocks, all N columns aligned at the output.
//   Only neighbour links inside the array; the single array-wide signal is w_shift (shadow chain enable).
//   The optional clock enable `en` of the contract is tied to 1 here (stub); the PEs already honour it.
module sa_core #(
  parameter int N         = 16,
  parameter int ACC_W     = 32,
  parameter int PE_TYPE   = 0,
  parameter bit ZERO_SKIP = 1'b1
) (
  input  logic                clk,
  input  logic                rst,
  input  logic                in_valid,
  input  logic                in_newW,
  input  logic [N*8-1:0]      in_vec,
  input  logic                w_shift,
  input  logic [N*8-1:0]      w_row,
  output logic                out_valid,
  output logic [N*ACC_W-1:0]  out_vec
);
  if (PE_TYPE != 0) begin : g_pe_type_check
    $error("sa_core: PE_TYPE=%0d not implemented (only PE_TYPE=0, math-block PEs)", PE_TYPE);
  end

  logic en;
  assign en = 1'b1;

  // ---------------- input skew: row r delayed by r clocks (invalid slots carry 0) ----------------
  logic signed [7:0] a_skew [N];
  logic              f_skew [N];

  for (genvar r = 0; r < N; r++) begin : g_skew
    logic signed [7:0] a_head;
    logic              f_head;
    assign a_head = in_valid ? in_vec[r*8 +: 8] : 8'sd0;
    assign f_head = in_valid & in_newW;
    if (r == 0) begin : g_none
      assign a_skew[r] = a_head;
      assign f_skew[r] = f_head;
    end else begin : g_dly
      localparam int D = r;
      logic signed [7:0] a_dly [D];
      logic              f_dly [D];
      always_ff @(posedge clk) begin
        if (rst) begin
          for (int k = 0; k < D; k++) begin
            a_dly[k] <= 8'sd0;
            f_dly[k] <= 1'b0;
          end
        end else if (en) begin
          a_dly[0] <= a_head;
          f_dly[0] <= f_head;
          for (int k = 1; k < D; k++) begin
            a_dly[k] <= a_dly[k-1];
            f_dly[k] <= f_dly[k-1];
          end
        end
      end
      assign a_skew[r] = a_dly[D-1];
      assign f_skew[r] = f_dly[D-1];
    end
  end

  // ---------------- PE grid: neighbour-only links ----------------
  logic signed [7:0]       a_link  [N][N+1];   // [r][c] = into PE(r,c); c == N = off the right edge
  logic                    f_link  [N][N+1];
  logic signed [ACC_W-1:0] ps_link [N+1][N];   // [r][c] = into PE(r,c); r == N = off the bottom edge
  logic signed [7:0]       sw_link [N+1][N];

  for (genvar r = 0; r < N; r++) begin : g_row
    assign a_link[r][0] = a_skew[r];
    assign f_link[r][0] = f_skew[r];
    for (genvar c = 0; c < N; c++) begin : g_col
      if (r == 0) begin : g_top
        assign ps_link[0][c] = '0;
        assign sw_link[0][c] = w_row[c*8 +: 8];
      end
      sa_pe #(.ACC_W(ACC_W), .ZERO_SKIP(ZERO_SKIP)) u_pe (
        .clk(clk), .rst(rst), .en(en),
        .a_in(a_link[r][c]),     .flag_in(f_link[r][c]),
        .ps_in(ps_link[r][c]),   .sw_in(sw_link[r][c]),   .sw_shift(w_shift),
        .a_out(a_link[r][c+1]),  .flag_out(f_link[r][c+1]),
        .ps_out(ps_link[r+1][c]), .sw_out(sw_link[r+1][c])
      );
    end
  end

  // ---------------- output de-skew: column c delayed N-1-c clocks ----------------
  for (genvar c = 0; c < N; c++) begin : g_deskew
    if (c == N-1) begin : g_none
      assign out_vec[c*ACC_W +: ACC_W] = ps_link[N][c];
    end else begin : g_dly
      localparam int D = N - 1 - c;
      logic signed [ACC_W-1:0] o_dly [D];
      always_ff @(posedge clk) begin
        if (rst) begin
          for (int k = 0; k < D; k++) o_dly[k] <= '0;
        end else if (en) begin
          o_dly[0] <= ps_link[N][c];
          for (int k = 1; k < D; k++) o_dly[k] <= o_dly[k-1];
        end
      end
      assign out_vec[c*ACC_W +: ACC_W] = o_dly[D-1];
    end
  end

  // ---------------- valid pipeline: out_valid rises 2N-1 clocks after in_valid was sampled ----------------
  logic [2*N-2:0] vpipe;
  always_ff @(posedge clk) begin
    if (rst)     vpipe <= '0;
    else if (en) vpipe <= {vpipe[2*N-3:0], in_valid};
  end
  assign out_valid = vpipe[2*N-2];
endmodule
