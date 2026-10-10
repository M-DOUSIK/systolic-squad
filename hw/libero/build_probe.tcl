# Synthesize + place&route + verify timing + generate programming data + export a FlashPro Express job (FABRIC + SNVM only:
# the HSS in eNVM, programmed from the stock job, is left untouched). Run after add_npu_slot.tcl:
#   libero SCRIPT:hw/libero/build_probe.tcl LOGFILE:C:/w/build_probe.log
# Output job: C:/w/out/MPFS_DISCOVERY.job (copied into hw/libero/out/ by hand after the build)
set proj {C:/w/ref/MPFS_DISCOVERY/MPFS_DISCOVERY.prjx}
set outdir {C:/w/out}
file mkdir $outdir

open_project -file $proj
source {C:/w/ref/script_support/additional_configurations/functions.tcl}

# same place&route options as the reference design script
configure_tool -name {PLACEROUTE} -params {DELAY_ANALYSIS:MAX} -params {EFFORT_LEVEL:true} -params {GB_DEMOTION:true} -params {INCRPLACEANDROUTE:false} -params {IOREG_COMBINING:false} -params {MULTI_PASS_CRITERIA:VIOLATIONS} -params {MULTI_PASS_LAYOUT:true} -params {NUM_MULTI_PASSES:5} -params {PDPR:false} -params {RANDOM_SEED:0} -params {REPAIR_MIN_DELAY:true} -params {REPLICATION:false} -params {SLACK_CRITERIA:WORST_SLACK} -params {SPECIFIC_CLOCK:} -params {START_SEED_INDEX:1} -params {STOP_ON_FIRST_PASS:true} -params {TDPR:true} 

# the edited SmartDesign (and its parent) must be generated before synthesis
generate_component -component_name {FIC_3_PERIPHERALS}
generate_component -component_name {MPFS_DISCOVERY_KIT}
build_design_hierarchy

run_tool -name {SYNTHESIZE}
run_tool -name {PLACEROUTE}
run_tool -name {VERIFYTIMING}
run_tool -name {GENERATEPROGRAMMINGDATA}

export_fpe_job MPFS_DISCOVERY $outdir "FABRIC SNVM" 0
save_project
puts "build_probe.tcl done"
