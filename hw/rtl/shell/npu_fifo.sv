// npu_fifo — synchronous show-ahead FIFO on an inferred simple dual-port RAM (LSRAM / uSRAM in Synplify).
//   dout is the head entry whenever empty == 0 (valid one clock after the push that filled an empty FIFO).
//   The RAM read is registered; a write to the address being read in the same clock is forwarded by a small bypass
//   register outside the RAM, so the RAM template stays inferable whatever its read-during-write behaviour is.
//   Push when full / pop when empty are ignored (the shell checks the flags first).
module npu_fifo #(
  parameter int W     = 32,
  parameter int DEPTH = 32                     // power of two
) (
  input  logic          clk,
  input  logic          rst,
  input  logic          push,
  input  logic [W-1:0]  din,
  input  logic          pop,
  output logic [W-1:0]  dout,
  output logic          empty,
  output logic          full,
  output logic [$clog2(DEPTH):0] count
);
  localparam int AW = $clog2(DEPTH);
  logic [W-1:0]  mem [DEPTH];
  logic [AW-1:0] wptr, rptr, raddr;
  logic          do_push, do_pop;
  logic [W-1:0]  rd_q, byp_d;
  logic          byp_sel;

  assign empty   = (count == 0);
  assign full    = (count == DEPTH);
  assign do_push = push && !full;
  assign do_pop  = pop && !empty;
  assign raddr   = do_pop ? rptr + 1'b1 : rptr;

  always_ff @(posedge clk) begin
    if (do_push) mem[wptr] <= din;
    rd_q <= mem[raddr];
  end

  always_ff @(posedge clk) begin
    byp_sel <= do_push && (wptr == raddr);
    byp_d   <= din;
  end
  assign dout = byp_sel ? byp_d : rd_q;

  always_ff @(posedge clk) begin
    if (rst) begin
      wptr  <= '0;
      rptr  <= '0;
      count <= '0;
    end else begin
      if (do_push) wptr <= wptr + 1'b1;
      if (do_pop)  rptr <= rptr + 1'b1;
      count <= count + (do_push ? 1'b1 : 1'b0) - (do_pop ? 1'b1 : 1'b0);
    end
  end
endmodule
