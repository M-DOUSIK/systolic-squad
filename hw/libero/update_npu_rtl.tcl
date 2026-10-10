# Put npu_top (+ sa_core, npu_postproc, npu_fifo) behind FIC3 slot 0 instead of apb_probe, then build.
# Run (Libero batch, after add_npu_slot.tcl was run once on C:/w/ref):
#   libero SCRIPT:hw/libero/update_npu_rtl.tcl LOGFILE:C:/w/update_npu.log
# Then build with build_probe.tcl (same flow, writes the job to C:/w/out).
set repo [file normalize [file join [file dirname [info script]] .. ..]]
set proj {C:/w/ref/MPFS_DISCOVERY/MPFS_DISCOVERY.prjx}
open_project -file $proj
foreach f {hw/rtl/core/sa_pe.sv hw/rtl/core/sa_core.sv hw/rtl/core/npu_postproc.sv hw/rtl/shell/npu_fifo.sv hw/rtl/shell/npu_top.sv} {
    set dst "C:/w/ref/MPFS_DISCOVERY/hdl/[file tail $f]"
    if {[file exists $dst]} { file copy -force "$repo/$f" $dst } else { import_files -library work -hdl_source "$repo/$f" }
}
file copy -force "$repo/hw/rtl/top/npu_apb_slot.v" {C:/w/ref/MPFS_DISCOVERY/hdl/npu_apb_slot.v}
build_design_hierarchy
save_project
close_project
puts "update_npu_rtl.tcl done"
