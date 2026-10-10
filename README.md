# Systolic Squad — Depth Estimation on a Systolic-Array NPU (PolarFire SoC)

IEEE EDS Edge AI Hackathon 2026 — FPGA track — PROC-05: 2D Systolic Array-Based Processing Element.

> 🏆 **First Prize, FPGA Track: IEEE EDS Edge AI Hackathon 2026**
>
> National-level finals at Parala Maharaja Engineering College (PMEC), Berhampur, Odisha, 4–6 October 2026, with 42 finalist teams. Awarded **₹20,000** and a **Renesas QuickConnect Beginners Kit V2.0**.

![FPGA](https://img.shields.io/badge/FPGA-PolarFire_SoC-E4002B)
![HDL](https://img.shields.io/badge/HDL-SystemVerilog-blue)
![NPU](https://img.shields.io/badge/NPU-16×16_INT8_systolic_array-informational)
![Award](https://img.shields.io/badge/🏆_1st_Prize-FPGA_Track,_IEEE_Edge_AI_Hackathon_2026-DAA520)
![Speed-up](https://img.shields.io/badge/speed--up-6.7×_vs_CPU-brightgreen)
![Python](https://img.shields.io/badge/ML-PyTorch_→_INT8-EE4C2C?logo=pytorch&logoColor=white)

## The project

We designed our own AI accelerator (an **NPU**): a **16×16 systolic array** of INT8 multiply-accumulate units, built in the FPGA fabric of
a **Microchip PolarFire SoC Discovery Kit**. The RISC-V processor on the same chip controls it and runs a real neural network on it.

## The application: depth from a single camera

Point any camera — the laptop webcam or a phone — at a scene, and the board produces a **depth map**: for every pixel, how far away it is
(red = near, blue = far). This is useful for robots and drones avoiding obstacles or as a navigation aid, and it runs on a small
low-power board instead of a GPU or the cloud.

Two networks run on the board, both converted to 8-bit integers. Switch between them with the **MODEL** button on the dashboard:

- **FastDepth** (MobileNet encoder + lightweight decoder): fast, and gives **distances in metres** (tap a pixel). We fine-tuned it for our
  128×96 input and trained it further with a much larger model (Depth Anything V2) as a teacher.
- **MiDaS v2.1 small** (EfficientNet-Lite3 encoder): slower but much cleaner depth maps; gives **relative** depth (near / far, no metres).

## How it works

```
camera (laptop / phone) ──> laptop ──(USB serial, 921600 baud)──> PolarFire SoC
                                                                   RISC-V U54 cores (bare-metal C)
                                                                     └─ APB bus ──> our NPU: 16×16 systolic array
                                                                                    + on-chip accumulator + input replay buffer
depth map on the screen <──────────────────────────────────────────────────────────┘     + requantiser
```

1. The laptop resizes each camera frame to 128×96 pixels and sends it to the board.
2. The CPU runs the network layer by layer. Every convolution is cut into 16×16 weight tiles and sent to the NPU; the array does
   256 multiply-adds per clock, adds up long dot products on chip and returns 8-bit results.
3. The NPU keeps the inputs of a layer in an on-chip **replay buffer** (128 KB): the CPU sends each input once, and the NPU replays it for
   every group of 16 output channels instead of the CPU sending it again. This halved the time of the NPU layers.
4. Depthwise convolutions, upsampling and skip connections run on the CPU, split over the 4 U54 cores.
5. The depth map goes back to the laptop and is shown in the web dashboard (laptop browser or phones via QR code).

## Results (measured on the board)

| | FastDepth | MiDaS small |
|---|---|---|
| Depth network with our NPU | **0.456 s** / frame | **1.99 s** / frame |
| Same frame on the CPU only | 3.05 s → NPU **6.7× faster** | 15.3 s → NPU **7.7× faster** |
| Whole frame (with the Canny change gate) | 0.61 s | 2.14 s |
| Output vs reference model | bit-identical | bit-identical |
| Size | 3.93 M weights, 180 M multiply-adds, 92 % on the NPU | 16.5 M weights, 862 M multiply-adds, 98 % on the NPU |
| Accuracy (NYU Depth v2 test set) | average error (RMSE) 0.78 m | relative depth (RMSE 0.54 m after scale alignment) |

| Hardware | |
|---|---|
| NPU self-test (NPU vs CPU) | 1000 / 1000 correct |
| FPGA clock / timing | 50 MHz, timing met (+3.20 ns slack) |
| FPGA usage | 55 % LUTs, 288 of 292 math blocks, 126 of 308 RAM blocks |

## What you need

- PolarFire SoC Discovery Kit + USB-C cable
- Windows laptop with Python 3 and: `pip install pyserial opencv-python pillow numpy qrcode`
- For programming the board: Microchip Libero SoC / SoftConsole (FlashPro Express and the boot-mode programmer come with them)

## Program the board (only once)

    powershell -File hw/libero/program_app.ps1 fw\board\out\depth_u54.elf -Job hw\libero\out\npu_fabric_only.job

This writes the FPGA design (our NPU) and the program into the chip's flash. Wait for `Chain programming PASSED` and
`mpfsBootmodeProgrammer completed successfully`, then unplug and replug the USB-C cable.

Both are kept when the power is off — the board starts the program by itself about 3 seconds after power-up. You only program again if the
FPGA design or the firmware changes.

## Run the demo

    python host/phone_server.py --port COM8

- The first time after power-up it sends the FastDepth weights to the board (4 MB, about 45 s). The board keeps them until it is switched off.
- The browser opens the dashboard at `http://localhost:8080`. Press **START CAMERA**.
- **MODEL** switches between FastDepth and MiDaS. The first switch to MiDaS sends its weights (17 MB, about 3 minutes, progress on the
  page); after that switching is instant until the board is switched off. To start with MiDaS: add `--model midas`.
- Phones: scan the QR code on the dashboard (same Wi-Fi as the laptop, or the laptop's mobile hotspot), accept the certificate warning once,
  press **START CAMERA**. On the laptop, **WATCH PHONES** shows what the phones send.
- **BENCH** runs the same frame on the CPU and on the NPU and shows the speed-up; **Self-test** checks the NPU against the CPU;
  **NPU: ON/OFF** switches the NPU off to compare live.
- Stop with Ctrl+C. Without a board: `python host/phone_server.py --emulate` (the laptop computes the same result).

## After a power cut

1. Plug the USB-C cable back in and wait about 5 seconds (the board starts the program by itself — no programming needed).
2. In PowerShell, in this folder: `python host/phone_server.py --port COM8`
3. Wait for the weight upload (about 45 s), then the dashboard opens at `http://localhost:8080`.

## See the board working (serial monitor)

While the demo runs, the board prints one line per frame on its second serial port. Open any serial terminal on
**COM10, 921600 baud, 8 data bits, no parity, 1 stop bit, no flow control** (text mode), or:

    python -m serial.tools.miniterm COM10 921600

```
model: FastDepth (MobileNet + NNConv5) (weights loaded)
frame 55 | NPU 16x16 | depth net 455 ms (array layers 355 ms, CPU layers 99 ms) | array busy 17 % of 22814310 clk | NPU errors 0 | frame total 607 ms
model: MiDaS v2.1 small (EfficientNet-Lite3) (weights loaded)
frame 56 | NPU 16x16 | depth net 1992 ms (array layers 1859 ms, CPU layers 131 ms) | array busy 28 % of 99645533 clk | NPU errors 0 | frame total 2143 ms
SELFTEST PASS: 1000/1000 NPU results (K=70 -> 5 K-tiles accumulated on chip) equal the CPU
BENCH same frame: CPU only 15276 ms, NPU 1991 ms, speed-up x7.6, outputs bit-identical
```
Don't type into the terminal — the board would start answering on that port.

## Troubleshooting

| Problem | Fix |
|---|---|
| `no PONG on COM8` | unplug the USB-C, wait 3 s, plug it back in, wait 5 s and run the command again (also try `--port COM10`) |
| Dashboard says CPU ONLY | the FPGA image is missing: program the board again (above) |
| "access denied" on COM8 | another program uses the port: close it |
| Weight upload CRC error | it retries automatically; if it keeps failing, replug the board |
| Phone can't open the page | same network as the laptop? allow Python in the Windows firewall |
| Need the previous NPU (without the replay buffer) | program with `-Job hw\libero\out\npu_fabric_v12_fallback.job` — the firmware detects it and still works, just slower |

## Project structure

```
hw/rtl/      the NPU in SystemVerilog: processing element, 16×16 array, requantiser, bus interface + FIFOs + accumulator + replay buffer
hw/tb/       testbenches and test vectors
hw/libero/   FPGA build scripts, ready FPGA images (out/npu_fabric_only.job, out/npu_fabric_v12_fallback.job), programming script
fw/          C firmware: NPU driver, network runtime (both models), serial protocol, edge gate; board/ = start-up code for the
             E51 + 4 U54 cores, build script, ready programs
ml/          training, INT8 conversion, reference integer model, export of the model files; export/ = FastDepth, export_midas/ = MiDaS
host/        dashboard + phone server, desktop app, serial protocol
reference/   Python reference of the integer maths that hardware and firmware match exactly
scripts/     simulation and test scripts
```

## Building from source

- **FPGA image**: Libero SoC 2026.1. Generate Microchip's Discovery Kit reference design, then run `hw/libero/add_npu_slot.tcl`,
  `hw/libero/update_npu_rtl.tcl` and `hw/libero/build_probe.tcl` (about 1.5 hours); copy the resulting `.job` to `hw/libero/out/`.
- **Firmware**: SoftConsole 2022.2 with Microchip's `mpfs-blank-baremetal` example: `powershell -File fw/board/build_app.ps1` → `fw/board/out/`.
- **Models**:
  - FastDepth: `python ml/export_all.py ml/export/fastdepth_distill2_128x96.pth` (PyTorch; datasets are downloaded with the scripts in `ml/data/`).
  - MiDaS: `python ml/convert_midas.py`, then `python ml/export_c.py ml/export_midas` and `python ml/make_vectors.py 2 ml/export_midas`
    (the MiDaS weights are downloaded automatically; needs `timm`).
- **Tests**: hardware `bash scripts/sim_shell.sh` (Icarus Verilog), firmware `make -C fw test` and `make -C fw test EXPORT=../ml/export_midas`,
  model files `python scripts/check_exports.py` (MiDaS: `python scripts/check_exports.py 0 export_midas`),
  board `python scripts/board_check.py COM8` (MiDaS: add `export_midas`).

## Credits

FastDepth (Wofk et al., ICRA 2019, MIT licence); starting weights from the community re-implementation Hagaik92/FastDepth; MiDaS v2.1 small
(Ranftl et al., TPAMI 2020, Intel ISL, MIT licence); Depth Anything V2 Small (Apache-2.0, used only as a teacher during training); NYU Depth v2
dataset; photos from Wikimedia Commons (licences in `ml/data/*/LICENSES.md`); Microchip PolarFire SoC Discovery Kit reference design and
bare-metal examples.

## License

Our code is released under the [MIT License](LICENSE). Third-party models and data keep their own licenses (see Credits).
