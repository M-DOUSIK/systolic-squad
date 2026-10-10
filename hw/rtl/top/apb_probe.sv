// apb_probe — minimal APB3 slave to prove the CPU -> FIC3 -> fabric path (bring-up only).
// 0x000 ID = 0x53514431 ("SQD1", RO), 0x044 SCRATCH (RW), 0x048 LEDS (RW, [7:0] drive leds output), others read 0.
// Same register offsets as the real npu_top, so the bring-up C code keeps working after the swap.
module apb_probe (
    input  logic        PCLK,
    input  logic        PRESETN,      // active-low, as on the FIC3 APB interface
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
    logic [31:0] scratch;
    logic [11:0] off;
    assign off     = PADDR[11:0];
    assign PREADY  = 1'b1;
    assign PSLVERR = 1'b0;

    always_ff @(posedge PCLK) begin
        if (!PRESETN) begin
            scratch <= 32'h0;
            leds    <= 8'h00;
        end else if (PSEL && PENABLE && PWRITE) begin
            if (off == 12'h044) scratch <= PWDATA;
            if (off == 12'h048) leds    <= PWDATA[7:0];
        end
    end

    always_comb begin
        case (off)
            12'h000: PRDATA = 32'h53514431;
            12'h044: PRDATA = scratch;
            12'h048: PRDATA = {24'h0, leds};
            default: PRDATA = 32'h0;
        endcase
    end
endmodule
