// npu_apb_slot - plain-Verilog wrapper that hosts our APB slave in FIC3 slot 0 (0x4000_0000, 256-byte window) of the
// Discovery Kit reference design (it replaces the unused PWM core). Libero's HDL+ core import wants Verilog; the real logic
// is SystemVerilog (apb_probe now, npu_top later), so this thin wrapper keeps the SmartDesign side simple.
// It hosts npu_top.
module npu_apb_slot (
    input         PCLK,
    input         PRESETN,
    input         PSEL,
    input         PENABLE,
    input         PWRITE,
    input  [31:0] PADDR,
    input  [31:0] PWDATA,
    output [31:0] PRDATA,
    output        PREADY,
    output        PSLVERR,
    output [7:0]  LEDS_OUT
);
    npu_top #(.N(16), .PE_TYPE(0), .ZERO_SKIP(1'b1), .ENGINE_ID(0), .ACC_DEPTH(1024), .XR_DEPTH(8192)) u_npu (
        .PCLK(PCLK), .PRESETN(PRESETN), .PSEL(PSEL), .PENABLE(PENABLE), .PWRITE(PWRITE),
        .PADDR(PADDR), .PWDATA(PWDATA), .PRDATA(PRDATA), .PREADY(PREADY), .PSLVERR(PSLVERR),
        .leds(LEDS_OUT)
    );
endmodule
