# Replace the (unused) PWM core in FIC3 slot 0 (0x4000_0000, 256 B) of the Discovery Kit reference design by our APB slave.
# Run (Libero batch):  libero SCRIPT:hw/libero/add_npu_slot.tcl LOGFILE:C:/w/add_slot.log   (after the reference design was generated in C:/w/ref)
# The slave is npu_apb_slot (Verilog wrapper) -> apb_probe now, npu_top later. Same offsets as
set repo   [file normalize [file join [file dirname [info script]] .. ..]]
set proj   {C:/w/ref/MPFS_DISCOVERY/MPFS_DISCOVERY.prjx}
set sd     {FIC_3_PERIPHERALS}

open_project -file $proj

# our HDL (SystemVerilog core + Verilog wrapper)
import_files -library work -hdl_source "$repo/hw/rtl/top/apb_probe.sv"
import_files -library work -hdl_source "$repo/hw/rtl/top/npu_apb_slot.v"
build_design_hierarchy

create_hdl_core -file {hdl/npu_apb_slot.v} -module {npu_apb_slot} -library {work} -package {}
hdl_core_add_bif -hdl_core_name {npu_apb_slot} -bif_definition {APB:AMBA:AMBA2:slave} -bif_name {APB_bif} -signal_map {\
"PADDR:PADDR" \
"PENABLE:PENABLE" \
"PWRITE:PWRITE" \
"PRDATA:PRDATA" \
"PWDATA:PWDATA" \
"PREADY:PREADY" \
"PSLVERR:PSLVERR" \
"PSELx:PSEL" }
build_design_hierarchy

open_smartdesign -sd_name $sd
sd_delete_instances -sd_name $sd -instance_names {PWM}
sd_instantiate_hdl_core -sd_name $sd -hdl_core_name {npu_apb_slot} -instance_name {NPU_0}
sd_connect_pins -sd_name $sd -pin_names {"FIC_3_ADDRESS_GENERATION_1:FIC_3_0x4000_00xx" "NPU_0:APB_bif"}
sd_connect_pins -sd_name $sd -pin_names {"NPU_0:PCLK" "PCLK"}
sd_connect_pins -sd_name $sd -pin_names {"NPU_0:PRESETN" "PRESETN"}
sd_mark_pins_unused -sd_name $sd -pin_names {NPU_0:LEDS_OUT}
sd_connect_pins_to_constant -sd_name $sd -pin_names {PWM_0} -value {GND}
save_smartdesign -sd_name $sd

build_design_hierarchy
save_project
close_project
puts "add_npu_slot.tcl done"
