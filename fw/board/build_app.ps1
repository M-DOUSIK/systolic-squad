# Build the depth application for the Discovery Kit (boot mode 1, eNVM).
# Usage (PowerShell):  powershell -File fw/board/build_app.ps1
#   -> fw/board/out/depth_u54.elf  (app loop on U54_1, which has an L1 data cache - the one we use)
#   -> fw/board/out/depth_e51.elf  (app loop on the E51 - fallback, slower CPU layers)
# Program fabric + app:  powershell -File hw/libero/program_app.ps1 fw\board\out\depth_u54.elf -Job hw\libero\out\npu_fabric_only.job
# The project is a copy of Microchip's mpfs-blank-baremetal example (config Discovery-Kit-eNVM-Scratchpad-Release, makefiles generated once by SoftConsole) in C:\w\probe;
# fw/board/e51.c and u54_1.c replace the example's hart files; e51.c includes the portable fw/src files, copied flat into src/application/hart0/fw/.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$sc   = 'C:\Microchip\SoftConsole-v2022.2-RISC-V-747'
$app  = 'C:\w\npuapp'
if (-not (Test-Path $app)) {
    Copy-Item 'C:\w\probe' $app -Recurse
    Get-ChildItem "$app\Discovery-Kit-eNVM-Scratchpad-Release" -Recurse -Include *.mk, makefile | ForEach-Object {
        (Get-Content $_.FullName -Raw).Replace('C:\w\probe', $app) | Set-Content $_.FullName -NoNewline
    }
}
$fwdst = "$app\src\application\hart0\fw"
New-Item -ItemType Directory -Force $fwdst | Out-Null
Copy-Item "$repo\fw\include\*.h" $fwdst -Force
$sfx = ''
foreach ($f in 'proto', 'qmath', 'npu', 'npu_conv3x3', 'canny', 'depth', 'models', 'obstacle', 'app') { Copy-Item "$repo\fw\src\$f.c" $fwdst -Force }
Copy-Item "$repo\fw\board\e51.c" "$app\src\application\hart0\e51.c" -Force
Copy-Item "$repo\fw\board\u54_1.c" "$app\src\application\hart1\u54_1.c" -Force
foreach ($n in 2, 3, 4) { Copy-Item "$repo\fw\board\u54_$n.c" "$app\src\application\hart$n\u54_$n.c" -Force }
$env:PATH = "$sc\riscv-unknown-elf-gcc\bin;$sc\build_tools\bin;$sc\python3;" + $env:PATH
$env:MACRO_PYTHON_BINARY_PATH_AND_EXECUTABLE = "$sc\python3\python.exe"
$ErrorActionPreference = 'Continue'      # compiler warnings go to stderr: not fatal
$out = "$repo\fw\board\out"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($v in @(@{hart = 0; name = "depth_e51$sfx.elf"}, @{hart = 1; name = "depth_u54$sfx.elf"})) {
    Set-Content "$app\src\application\hart0\board_cfg.h" "#define APP_HART $($v.hart)" -Encoding ascii
    (Get-Item "$app\src\application\hart0\e51.c").LastWriteTime = Get-Date        # board_cfg.h is new: force both harts to rebuild
    (Get-Item "$app\src\application\hart1\u54_1.c").LastWriteTime = Get-Date
    foreach ($n in 2, 3, 4) { (Get-Item "$app\src\application\hart$n\u54_$n.c").LastWriteTime = Get-Date }
    Remove-Item "$out\$($v.name)" -ErrorAction SilentlyContinue                    # never leave a stale ELF behind
    Remove-Item "$app\Discovery-Kit-eNVM-Scratchpad-Release\mpfs-blank-baremetal.elf" -ErrorAction SilentlyContinue
    Push-Location "$app\Discovery-Kit-eNVM-Scratchpad-Release"
    & make -j8 all 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match ' error|error:|undefined reference|overflowed' } | Select-Object -Last 20
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { "BUILD FAILED for $($v.name)"; exit 1 }
    Copy-Item "$app\Discovery-Kit-eNVM-Scratchpad-Release\mpfs-blank-baremetal.elf" "$out\$($v.name)" -Force
    $sz = & "$sc\riscv-unknown-elf-gcc\bin\riscv64-unknown-elf-size.exe" "$out\$($v.name)"
    "$($v.name): $($sz[-1])"
}
