# Program our fabric job and a bare-metal app (boot mode 1, eNVM) on the Discovery Kit. Usage: powershell -File hw/libero/program_app.ps1 <app.elf>
# Afterwards power-cycle the board (unplug/replug USB-C). Restore the stock HSS design with hw/libero/program_stock.tcl.
param([Parameter(Mandatory=$true)][string]$Elf, [string]$Job = 'hw\libero\out\probe_fabric_only.job')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$lib  = 'C:\Microchip\Libero_SoC_2026.1\Libero_SoC\Designer'
$sc   = 'C:\Microchip\SoftConsole-v2022.2-RISC-V-747'
$env:LM_LICENSE_FILE = 'C:\Microchip\License\License.dat'
# 1. fabric (+ sNVM), HSS/eNVM untouched
$proj = 'C:\w\fpe_fab'; New-Item -ItemType Directory -Force $proj | Out-Null
$job  = (Join-Path $repo $Job) -replace '\\','/'
$tcl  = "$env:TEMP\program_fabric.tcl"
Set-Content $tcl "create_job_project -job_project_location {C:/w/fpe_fab} -job_file {$job} -overwrite 1`nset_programming_action -name {MPFS095T} -action {PROGRAM}`nrun_selected_actions`nsave_project`nclose_project"
& "$lib\bin\FPExpress.exe" "SCRIPT:$tcl" "LOGFILE:C:\w\program_fabric.log" | Out-Null
Select-String -Path C:\w\program_fabric.log -Pattern 'PASSED|FAILED' | Select -Last 1 | ForEach-Object { $_.Line }
# 2. app -> eNVM, boot mode 1 (Microchip's boot-mode programmer adds the boot header)
$env:SC_INSTALL_DIR = $sc
$env:FPGENPROG = "$lib\bin64\fpgenprog.exe"
$wd = 'C:\w\bootmode'; New-Item -ItemType Directory -Force $wd | Out-Null
Copy-Item $Elf "$wd\app.elf" -Force
Push-Location $wd
& "$sc\eclipse\jre\bin\java.exe" -jar "$sc\extras\mpfs\mpfsBootmodeProgrammer.jar" --workdir $wd --die MPFS095T --package FCSG325 --bootmode 1 app.elf
Pop-Location
'Now unplug and replug the USB-C cable.'
