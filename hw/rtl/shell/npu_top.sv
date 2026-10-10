// npu_top — APB shell around sa_core + K-tile accumulator + npu_postproc.
//
//   APB write W_DATA (N/4 words) -> one weight row -> w_shift            (rows N-1 .. 0, then W_COMMIT)
//   APB write X_DATA (N/4 words) -> IN FIFO {newW, first, last, slot, vec}
//   issue (credit rule) -> sa_core (latency 2N-1) -> accumulator (2 clocks) -> npu_postproc (3 clocks, only LAST vectors)
//   -> OUT FIFO (INT32 x N in RAW mode, INT8 x N packed in RQ mode) -> APB read Y_DATA
//
// K-tile accumulator : with CTRL.ACC_EN = 1 every pushed vector carries the FIRST / LAST flags of the
// current pass (register ACC_PASS) and a slot number = its position in the pass (the slot counter restarts at every ACC_PASS
// write). acc[slot] = (FIRST ? 0 : acc[slot]) + y. Only LAST vectors go through postproc into the OUT FIFO. With ACC_EN = 0
// every vector counts as FIRST and LAST, i.e. the v1.1 behaviour (2 clocks more latency).
//
// X replay buffer (v1.3): with CTRL.XR_REC = 1 every X vector written over APB is also stored in xr_mem[XR_WPTR++]. A write to
// XR_REPLAY {count[31:16], start[15:0]} makes the shell push xr_mem[start .. start+count-1] into the IN FIFO itself, one per clock,
// through the same tagging path as APB vectors (newW, FIRST/LAST, slot). The driver records a chunk's X once (first C_out tile) and
// replays it for every other C_out tile: the APB link carries each input vector once instead of once per 16 output channels.
// Only neighbour links inside sa_core; the shell is ordinary control logic. One clock (PCLK), synchronous reset.
module npu_top #(
  parameter int N         = 16,
  parameter int PE_TYPE   = 0,
  parameter bit ZERO_SKIP = 1'b1,
  parameter int ENGINE_ID = 0,
  parameter int ACC_DEPTH = 1024,              // accumulator slots (vectors per pass); power of two
  parameter int XR_DEPTH  = 256,               // X replay buffer (vectors); power of two, <= 32768
  parameter int IN_DEPTH  = 32,
  parameter int OUT_DEPTH = 64
) (
  input  logic        PCLK,
  input  logic        PRESETN,
  input  logic        PSEL,
  input  logic        PENABLE,
  input  logic        PWRITE,
  input  logic [31:0] PADDR,
  input  logic [31:0] PWDATA,
  output logic [31:0] PRDATA,
  output logic        PREADY,
  output logic        PSLVERR,
  output logic [7:0]  leds
);
  localparam int ACC_W   = 32;
  localparam int NW      = N / 4;              // 32-bit words per INT8 vector
  localparam int SLOT_W  = $clog2(ACC_DEPTH);
  localparam int LAT     = 2 * N - 1;          // sa_core latency (contract)
  localparam int TAG_W   = 2 + SLOT_W;         // {first, last, slot}
  localparam int INW     = 1 + TAG_W + N * 8;  // {newW, tag, vec}
  localparam int XR_W    = $clog2(XR_DEPTH);
  localparam logic [7:0] VERSION = 8'd3;

  logic clk, rst_hw, srst, rst;
  assign clk    = PCLK;
  assign rst_hw = !PRESETN;
  assign rst    = rst_hw | srst;
  assign PREADY  = 1'b1;
  assign PSLVERR = 1'b0;

  // ------------------------------------------------------------------------------------------------ APB decode
  logic [7:0] off;
  logic       wr, rd;
  assign off = PADDR[7:0];
  assign wr  = PSEL && PENABLE && PWRITE;
  assign rd  = PSEL && PENABLE && !PWRITE;

  // ------------------------------------------------------------------------------------------------ registers
  logic        rq_en, relu_en, sleep, zskip_en, acc_en, xr_rec;
  logic [31:0] scratch;
  logic signed [7:0] out_zp;
  logic [7:0]  col_idx;
  logic [15:0] mult_r  [N];
  logic [4:0]  shift_r [N];
  logic [31:0] bias_r  [N];
  logic        err_w, err_ovf, err_unf, err_acc;
  logic        perf_en;
  logic [31:0] perf_cycles, perf_active, perf_vectors, perf_zero, perf_sleep;
  logic        pass_first, pass_last;
  logic [SLOT_W:0] slot_ctr;

  // ------------------------------------------------------------------------------------------------ status signals
  logic [$clog2(IN_DEPTH):0]  in_count;
  logic [$clog2(OUT_DEPTH):0] out_count;
  logic in_empty, in_full, out_empty, out_full;
  logic [7:0] inflight;                        // issued, not yet in the OUT FIFO or dropped (non-LAST)
  logic busy, w_ready;

  // ------------------------------------------------------------------------------------------------ weight path
  logic [N*8-1:0] wbuf;
  logic [$clog2(NW+1)-1:0] widx;
  logic        w_shift;
  logic [N*8-1:0] w_row;
  logic        commit_pending;
  logic [7:0]  newW_in_fifo;                   // newW vectors waiting in the IN FIFO
  logic [7:0]  wblock;                         // clocks left in which no w_shift is allowed

  logic [N*8-1:0] wbuf_ins, xbuf_ins;            // assembler buffer with the word being written inserted
  assign w_ready = !commit_pending && (newW_in_fifo == 0) && (wblock == 0);

  // ------------------------------------------------------------------------------------------------ X path / IN FIFO
  logic [N*8-1:0] xbuf;
  logic [$clog2(NW+1)-1:0] xidx;
  logic          in_push, in_pop;
  logic [INW-1:0] in_din, in_dout;
  logic          apb_vec;                      // the APB word that completes an X vector

  // ------------------------------------------------------------------------------------------------ X replay buffer
  logic [N*8-1:0] xr_mem [XR_DEPTH];
  logic [N*8-1:0] xr_q;
  logic [XR_W:0]  xr_wptr;
  logic           xr_we;
  logic [XR_W-1:0] rp_addr;
  logic [XR_W:0]  rp_left;
  logic           rp_issue, rp_rd, rp_active;

  // ------------------------------------------------------------------------------------------------ OUT FIFO / Y path
  logic                 out_push, out_pop;
  logic [N*32-1:0]      out_din, out_dout;
  logic [$clog2(N+1)-1:0] yidx;
  logic [31:0]          y_word;
  logic                 y_last_word;
  assign y_word      = out_dout[32 * yidx +: 32];
  assign y_last_word = rq_en ? (yidx == NW - 1) : (yidx == N - 1);

  // ================================================================================================ register writes
  always_ff @(posedge clk) begin
    srst    <= 1'b0;
    if (rst_hw) begin
      rq_en <= 1'b0; relu_en <= 1'b0; sleep <= 1'b0; zskip_en <= 1'b1; acc_en <= 1'b0; xr_rec <= 1'b0;
      scratch <= '0; leds <= '0; out_zp <= '0; col_idx <= '0;
      for (int c = 0; c < N; c++) begin mult_r[c] <= 16'd1; shift_r[c] <= '0; bias_r[c] <= '0; end
      perf_en <= 1'b0;
    end else if (wr) begin
      case (off)
        8'h08: begin
          srst     <= PWDATA[0];
          rq_en    <= PWDATA[1];
          relu_en  <= PWDATA[2];
          sleep    <= PWDATA[3];
          zskip_en <= PWDATA[4];
          acc_en   <= PWDATA[5];
          xr_rec   <= PWDATA[6];
        end
        8'h20: mult_r[col_idx[$clog2(N)-1:0]]  <= PWDATA[15:0];
        8'h24: shift_r[col_idx[$clog2(N)-1:0]] <= PWDATA[4:0];
        8'h28: col_idx <= PWDATA[7:0];
        8'h2C: begin bias_r[col_idx[$clog2(N)-1:0]] <= PWDATA; col_idx <= col_idx + 8'd1; end
        8'h3C: perf_en <= PWDATA[1];
        8'h44: scratch <= PWDATA;
        8'h48: leds    <= PWDATA[7:0];
        8'h4C: out_zp  <= PWDATA[7:0];
        default: ;
      endcase
    end
  end

  // ================================================================================================ weight row assembler
  always_comb begin
    wbuf_ins = wbuf; wbuf_ins[32 * widx +: 32] = PWDATA;
    xbuf_ins = xbuf; xbuf_ins[32 * xidx +: 32] = PWDATA;
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      widx <= '0; wbuf <= '0; w_row <= '0; commit_pending <= 1'b0; err_w <= 1'b0; w_shift <= 1'b0;
    end else begin
      w_shift <= 1'b0;
      if (wr && off == 8'h40 && PWDATA[24]) err_w <= 1'b0;
      if (wr && off == 8'h10) begin
        if (!w_ready) err_w <= 1'b1;                                       // dropped
        else begin
          wbuf[32 * widx +: 32] <= PWDATA;
          if (widx == NW - 1) begin
            widx    <= '0;
            w_row   <= wbuf_ins;
            w_shift <= 1'b1;
          end else widx <= widx + 1'b1;
        end
      end
      if (wr && off == 8'h14) commit_pending <= 1'b1;
      if (in_push && in_din[INW-1]) commit_pending <= 1'b0;
    end
  end

  // ================================================================================================ X vector assembler / replay -> IN FIFO
  // A vector comes either from the APB assembler or from the replay buffer (rp_rd); both get the same tags. The driver never
  // writes X_DATA while a replay runs; if it does, the APB vector is dropped and ERR_OVF is set.
  assign apb_vec = wr && off == 8'h18 && xidx == NW - 1;
  always_ff @(posedge clk) begin
    if (rst) begin
      xidx <= '0; xbuf <= '0; err_ovf <= 1'b0; err_acc <= 1'b0;
      slot_ctr <= '0; pass_first <= 1'b1; pass_last <= 1'b1;
      in_din <= '0; in_push <= 1'b0;
    end else begin
      in_push <= 1'b0;
      if (wr && off == 8'h40) begin
        if (PWDATA[25]) err_ovf <= 1'b0;
        if (PWDATA[27]) err_acc <= 1'b0;
      end
      if (wr && off == 8'h58) begin
        pass_first <= PWDATA[0];
        pass_last  <= PWDATA[1];
        slot_ctr   <= '0;
      end
      if (wr && off == 8'h18) begin
        if (xidx == NW - 1) xidx <= '0;
        else begin
          xbuf[32 * xidx +: 32] <= PWDATA;
          xidx <= xidx + 1'b1;
        end
      end
      if (rp_rd || apb_vec) begin
        if (in_full || (rp_rd && apb_vec)) err_ovf <= 1'b1;               // dropped
        if (!in_full) begin
          in_din <= {commit_pending,
                     (acc_en ? pass_first : 1'b1), (acc_en ? pass_last : 1'b1), slot_ctr[SLOT_W-1:0],
                     (rp_rd ? xr_q : xbuf_ins)};
          in_push <= 1'b1;
          if (acc_en) begin
            if (slot_ctr == ACC_DEPTH) err_acc <= 1'b1;
            else slot_ctr <= slot_ctr + 1'b1;
          end
        end
      end
    end
  end

  // ================================================================================================ X replay buffer
  // record: APB vectors that enter the IN FIFO while XR_REC = 1 (and no replay runs) go to xr_mem[xr_wptr++]
  // replay: one read per clock while the IN FIFO has room for the reads in flight (issue -> read -> push -> count: 3 clocks)
  assign xr_we     = apb_vec && !in_full && !rp_rd && xr_rec && (32'(xr_wptr) < XR_DEPTH);
  assign rp_issue  = (rp_left != 0) && (32'(in_count) < IN_DEPTH - 4);
  assign rp_active = (rp_left != 0) || rp_rd || in_push;
  always_ff @(posedge clk) begin
    if (xr_we) xr_mem[xr_wptr[XR_W-1:0]] <= xbuf_ins;
    xr_q <= xr_mem[rp_addr];
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      xr_wptr <= '0; rp_addr <= '0; rp_left <= '0; rp_rd <= 1'b0;
    end else begin
      rp_rd <= rp_issue;
      if (wr && off == 8'h60)  xr_wptr <= PWDATA[XR_W:0];
      else if (xr_we)          xr_wptr <= xr_wptr + 1'b1;
      if (wr && off == 8'h64) begin
        rp_addr <= PWDATA[XR_W-1:0];
        rp_left <= PWDATA[16 +: XR_W + 1];
      end else if (rp_issue) begin
        rp_addr <= rp_addr + 1'b1;
        rp_left <= rp_left - 1'b1;
      end
    end
  end

  npu_fifo #(.W(INW), .DEPTH(IN_DEPTH)) u_in (
    .clk(clk), .rst(rst), .push(in_push), .din(in_din), .pop(in_pop), .dout(in_dout),
    .empty(in_empty), .full(in_full), .count(in_count));

  // ================================================================================================ issue (credit rule)
  logic [$clog2(OUT_DEPTH):0] out_free;
  logic issue;
  assign out_free = OUT_DEPTH[$clog2(OUT_DEPTH):0] - out_count;
  assign issue    = !in_empty && !sleep && (16'(inflight) < 16'(out_free));
  assign in_pop   = issue;

  logic             core_valid, core_newW;
  logic [N*8-1:0]   core_vec;
  logic [TAG_W-1:0] core_tag;
  assign core_valid = issue;
  assign core_newW  = in_dout[INW-1];
  assign core_tag   = in_dout[N*8 +: TAG_W];
  assign core_vec   = in_dout[N*8-1:0];

  // newW bookkeeping and the w_shift block window (2N clocks after a newW vector is issued)
  always_ff @(posedge clk) begin
    if (rst) begin
      newW_in_fifo <= '0;
      wblock       <= '0;
    end else begin
      newW_in_fifo <= newW_in_fifo + ((in_push && in_din[INW-1]) ? 8'd1 : 8'd0) - ((issue && core_newW) ? 8'd1 : 8'd0);
      if (issue && core_newW) wblock <= 8'(2 * N);
      else if (wblock != 0)   wblock <= wblock - 8'd1;
    end
  end

  logic             sa_out_valid;
  logic [N*32-1:0]  sa_out_vec;
  sa_core #(.N(N), .ACC_W(ACC_W), .PE_TYPE(PE_TYPE), .ZERO_SKIP(ZERO_SKIP)) u_core (
    .clk(clk), .rst(rst), .in_valid(core_valid), .in_newW(core_newW), .in_vec(core_vec),
    .w_shift(w_shift), .w_row(w_row), .out_valid(sa_out_valid), .out_vec(sa_out_vec));

  // tags travel next to the array in a delay line of exactly the array latency (the contract fixes it at 2N-1)
  logic [TAG_W:0] tag_dly [LAT];
  always_ff @(posedge clk) begin
    for (int i = 0; i < LAT; i++) begin
      if (rst) tag_dly[i] <= '0;
      else     tag_dly[i] <= (i == 0) ? {core_valid, core_tag} : tag_dly[i - 1];
    end
  end
  logic             t_valid, t_first, t_last;
  logic [SLOT_W-1:0] t_slot;
  assign {t_valid, t_first, t_last, t_slot} = tag_dly[LAT - 1];

  // ================================================================================================ K-tile accumulator
  logic [N*32-1:0]   acc_mem [ACC_DEPTH];
  logic [N*32-1:0]   acc_rd;
  logic              s1_valid, s1_first, s1_last;
  logic [SLOT_W-1:0] s1_slot;
  logic [N*32-1:0]   s1_y, s1_old, s1_sum;
  logic              wq_en;
  logic [SLOT_W-1:0] wq_addr;
  logic [N*32-1:0]   wq_data;

  always_ff @(posedge clk) begin
    acc_rd <= acc_mem[t_slot];
    if (s1_valid) acc_mem[s1_slot] <= s1_sum;
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      s1_valid <= 1'b0; wq_en <= 1'b0;
    end else begin
      s1_valid <= sa_out_valid;
      wq_en    <= s1_valid;
    end
    s1_first <= t_first; s1_last <= t_last; s1_slot <= t_slot; s1_y <= sa_out_vec;
    wq_addr  <= s1_slot; wq_data <= s1_sum;
  end
  // read issued in the same clock as the previous vector's write: forward that write
  assign s1_old = (wq_en && wq_addr == s1_slot) ? wq_data : acc_rd;
  for (genvar c = 0; c < N; c++) begin : g_acc
    assign s1_sum[32 * c +: 32] = (s1_first ? 32'd0 : s1_old[32 * c +: 32]) + s1_y[32 * c +: 32];
  end

  // ================================================================================================ postproc -> OUT FIFO
  logic [N*16-1:0] pp_mult;
  logic [N*5-1:0]  pp_shift;
  logic [N*32-1:0] pp_bias;
  for (genvar c = 0; c < N; c++) begin : g_pp
    assign pp_mult[16 * c +: 16] = mult_r[c];
    assign pp_shift[5 * c +: 5]  = shift_r[c];
    assign pp_bias[32 * c +: 32] = bias_r[c];
  end
  logic            pp_valid;
  logic [N*32-1:0] pp_raw;
  logic [N*8-1:0]  pp_q;
  npu_postproc #(.N(N)) u_pp (
    .clk(clk), .rst(rst), .in_valid(s1_valid && s1_last), .in_acc(s1_sum), .rq_en(rq_en), .relu_en(relu_en),
    .rq_mult(pp_mult), .rq_shift(pp_shift), .bias(pp_bias), .out_zp(out_zp),
    .out_valid(pp_valid), .out_raw(pp_raw), .out_q(pp_q));

  assign out_push = pp_valid;
  assign out_din  = rq_en ? {{(N * 24){1'b0}}, pp_q} : pp_raw;
  npu_fifo #(.W(N * 32), .DEPTH(OUT_DEPTH)) u_out (
    .clk(clk), .rst(rst), .push(out_push), .din(out_din), .pop(out_pop), .dout(out_dout),
    .empty(out_empty), .full(out_full), .count(out_count));

  // in-flight count for the credit rule: +1 per issue, -1 when a non-LAST vector leaves the accumulator, -1 per OUT push
  always_ff @(posedge clk) begin
    if (rst) inflight <= '0;
    else inflight <= inflight + (issue ? 8'd1 : 8'd0) - ((s1_valid && !s1_last) ? 8'd1 : 8'd0) - (out_push ? 8'd1 : 8'd0);
  end
  assign busy = !in_empty || (inflight != 0) || w_shift || rp_active;

  // ================================================================================================ Y_DATA reads
  assign out_pop = rd && off == 8'h1C && !out_empty && y_last_word;
  always_ff @(posedge clk) begin
    if (rst) begin
      yidx <= '0; err_unf <= 1'b0;
    end else begin
      if (wr && off == 8'h40 && PWDATA[26]) err_unf <= 1'b0;
      if (rd && off == 8'h1C) begin
        if (out_empty) err_unf <= 1'b1;
        else yidx <= y_last_word ? '0 : yidx + 1'b1;
      end
    end
  end

  // ================================================================================================ perf counters (kept by SOFT_RST)
  logic [7:0] core_occ;                                  // vectors inside sa_core
  logic [$clog2(N+1)-1:0] zcount;
  always_comb begin
    zcount = '0;
    for (int r = 0; r < N; r++) zcount = zcount + (core_vec[8 * r +: 8] == 8'd0 ? 1'b1 : 1'b0);
  end
  always_ff @(posedge clk) begin
    if (rst_hw) begin
      core_occ <= '0;
      perf_cycles <= '0; perf_active <= '0; perf_vectors <= '0; perf_zero <= '0; perf_sleep <= '0;
    end else begin
      core_occ <= srst ? 8'd0 : core_occ + (issue ? 8'd1 : 8'd0) - (sa_out_valid ? 8'd1 : 8'd0);
      if (wr && off == 8'h3C && PWDATA[0]) begin
        perf_cycles <= '0; perf_active <= '0; perf_vectors <= '0; perf_zero <= '0; perf_sleep <= '0;
      end else if (perf_en) begin
        perf_cycles <= perf_cycles + 1;
        if (core_occ != 0 || issue) perf_active <= perf_active + 1;
        if (issue) begin
          perf_vectors <= perf_vectors + 1;
          perf_zero    <= perf_zero + zcount;
        end
        if (sleep) perf_sleep <= perf_sleep + 1;
      end
    end
  end

  // ================================================================================================ read mux
  logic [31:0] status;
  assign status = {4'd0, err_acc, err_unf, err_ovf, err_w, 3'd0, rp_active, out_empty, in_full, w_ready, busy,
                   8'(out_count), 8'(in_count)};
  always_comb begin
    case (off)
      8'h00: PRDATA = 32'h53514431;
      8'h04: PRDATA = {VERSION, 3'd0, 3'(ENGINE_ID), ZERO_SKIP, 1'(PE_TYPE), 8'(ACC_W), 8'(N)};
      8'h08: PRDATA = {25'd0, xr_rec, acc_en, zskip_en, sleep, relu_en, rq_en, 1'b0};
      8'h0C: PRDATA = status;
      8'h1C: PRDATA = out_empty ? 32'hDEADBEEF : y_word;
      8'h20: PRDATA = {16'd0, mult_r[col_idx[$clog2(N)-1:0]]};
      8'h24: PRDATA = {27'd0, shift_r[col_idx[$clog2(N)-1:0]]};
      8'h28: PRDATA = {24'd0, col_idx};
      8'h2C: PRDATA = bias_r[col_idx[$clog2(N)-1:0]];
      8'h30: PRDATA = perf_cycles;
      8'h34: PRDATA = perf_active;
      8'h38: PRDATA = perf_vectors;
      8'h44: PRDATA = scratch;
      8'h48: PRDATA = {24'd0, leds};
      8'h4C: PRDATA = {24'd0, out_zp};
      8'h50: PRDATA = perf_zero;
      8'h54: PRDATA = perf_sleep;
      8'h58: PRDATA = {30'd0, pass_last, pass_first};
      8'h5C: PRDATA = ACC_DEPTH;
      8'h60: PRDATA = 32'(xr_wptr);
      8'h64: PRDATA = 32'(rp_left);
      8'h68: PRDATA = XR_DEPTH;
      default: PRDATA = 32'd0;
    endcase
  end
endmodule
