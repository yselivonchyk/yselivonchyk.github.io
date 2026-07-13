
<!-- 
Article purpose:
internalize the parallelism model of Hopper architecture

Not a purpose:
- no memory structure
- no compilation

Article structure
Basics 101: stream and kernel execution
Basics 102: stream and kernel resources
Multi-stream
resource vacancy
Advanced A: tensor cores
Advanced B: TMA

 -->
# CUDA Concurrency and GPU Resource Cheat Sheet

## 1. Execution hierarchy

```text
GPU
 ├── many SMs
 │    ├── multiple resident thread blocks
 │    │    └── multiple warps
 │    │         └── 32 threads per warp
```

A **thread block lives entirely on one SM**. Multiple blocks, including blocks from different concurrently executing kernels, may reside on the same SM.

Warps are the units selected for instruction issue. Resident warps keep their execution state on-chip, allowing the SM to switch between ready warps essentially without context-switch overhead.

---

## 2. Resource hierarchy — H100 example

### Per SM

An H100 SM can support, subject to other resource constraints:

* **32 resident thread blocks**
* **64 resident warps**
* **2,048 resident threads**
* **65,536 32-bit registers**
* **228 KB shared memory**
* Warp schedulers and execution pipelines for FP32, INT, Tensor Core, load/store, etc.

These resources determine whether another block can become resident on that SM.

Example:

```text
SM 0:
    Kernel A block
    Kernel A block
    Kernel B block
```

This is possible when enough registers, shared memory, thread/warp slots, and block slots remain.

### Per block

A block owns:

* its allocated shared-memory region;
* its threads and warps;
* block-level synchronization state.

Shared memory is allocated per block from the finite shared-memory capacity of its SM.

### Per thread

Each thread consumes registers. For example:

```text
256 threads/block × 64 registers/thread
= 16,384 registers/block
```

With 65,536 registers per SM, registers alone would limit residency to four such blocks per SM.

### Shared more broadly across the GPU

Across the GPU, kernels and streams may contend for:

* all available SMs;
* L2 cache;
* HBM capacity and bandwidth;
* memory controllers and internal memory fabric;
* device interconnect such as NVLink/PCIe paths;
* dedicated copy engines.

**TMA is different:** Hopper's Tensor Memory Accelerator is used by kernels for asynchronous memory movement involving shared memory. It should not be thought of as a separate device-wide engine through which an otherwise blocked stream can arbitrarily advance.

---

## 3. What streams mean

A CUDA stream is an **ordered lineage of operations**:

```text
Kernel A → copy → Kernel B → Kernel C
```

Operations within one stream preserve order. Therefore, Kernel B does not begin before earlier dependent work in that stream has completed.

Different streams permit independent work to overlap when the required hardware resources are available.

---

## 4. Compute concurrency between streams

Generally, one stream may launch a kernel that places work on every SM.

### Case 1: Some SMs are vacant

Another stream may place blocks on those vacant SMs:

```text
SM 0: Kernel A
SM 1: Kernel A
SM 2: vacant    → Kernel B can run here
SM 3: vacant    → Kernel B can run here
```

### Case 2: Every SM is occupied, yet spare residency capacity remains

Another kernel may still run concurrently by placing blocks on already occupied SMs:

```text
SM 0: [A block] [A block] [B block]
SM 1: [A block] [B block]
```

This requires enough remaining:

* thread/warp slots;
* registers;
* shared memory;
* block slots.

Therefore:

> **All SMs occupied does not imply another kernel must wait. A second kernel only needs enough residency capacity somewhere to place at least one block.**

### Case 3: Every SM is occupied and no additional block can become resident

Then another compute kernel waits, even when the currently resident kernel leaves execution pipelines idle.

For example:

```text
64 / 64 warp slots occupied
all A warps waiting on memory
FP32 units idle
issue slots idle

Kernel B still cannot run
because no B block can become resident.
```

---

## 5. Long memory waits and co-occupancy

A memory-bound kernel may occupy every SM while many of its warps spend substantial time waiting for data.

Example:

```text
Warp A: global load ───────────── waiting ─────────────→ ready
Warp B:                     FP32 instruction
Warp C:                          INT instruction
```

The SM scheduler issues instructions from other ready resident warps while Warp A waits. This is ordinary latency hiding.

A second kernel can help fill those idle issue opportunities **only when some of its blocks can also become resident**. Once resident, its warps join the pool of eligible warps:

```text
Resident warps:
    A0 A1 A2 A3 ...   from Kernel A
    B0 B1 B2 ...      from Kernel B

Scheduler:
    issue A0
    issue B1
    issue B2
    issue A3
```

So the crucial distinction is:

> **Unused execution capacity can be exploited only by already-resident warps. Spare issue slots alone do not allow another kernel to run; that kernel first needs residency resources.**

---

## 6. Tensor Core co-occupancy

A kernel heavily using Tensor Cores may leave other execution resources underutilized, such as some FP32, INT, or load/store capacity.

Another co-resident kernel could theoretically exploit complementary execution resources. For example:

```text
Kernel A: Tensor Core-heavy
Kernel B: INT-heavy
```

However, both kernels still share several resources:

* warp schedulers and issue capacity;
* register file;
* shared memory;
* L1/L2;
* HBM bandwidth.

Therefore, complementary execution units do not guarantee useful speedup. The second kernel first needs room to become resident, and then the two kernels must avoid contending for the same bottleneck.

The practical rule is:

> **Residency determines whether concurrency is possible. Execution-pipeline and memory contention determine whether that concurrency is useful.**

---

## 7. Useful work outside SM compute

Even when no second compute kernel can become resident, another stream may sometimes advance work using hardware outside ordinary SM compute execution.

Examples:

* asynchronous host-to-device copies through dedicated copy engines;
* asynchronous device-to-host copies;
* some peer-to-peer or NVLink transfers, depending on topology and operation.

Thus:

```text
Stream A:
    large compute kernel using all SM residency

Stream B:
    another compute kernel    → waits
    async H2D/D2H copy        → may advance concurrently
```

The exact overlap depends on hardware support and whether the operations contend for shared resources such as HBM bandwidth.

---

## 8. Final mental model

```text
Can another stream do useful work?
        |
        +-- Another compute kernel?
        |       |
        |       +-- Can one of its blocks become resident?
        |               |
        |               +-- Yes → it may execute concurrently
        |               +-- No  → it waits
        |
        +-- Copy or other independent hardware operation?
                |
                +-- May proceed concurrently, subject to
                    hardware support and shared bottlenecks
```

The shortest useful summary is:

> **Blocks are scheduled onto SMs. Multiple kernels may share one SM. Registers, shared memory, thread/warp slots, and block slots govern residency. Resident warps compete for issue opportunities and execution pipelines. L2, HBM, interconnects, and some transfer engines are shared more broadly. Another stream can execute compute whenever one of its blocks fits somewhere; otherwise only work using independent hardware, such as certain asynchronous copies, may still advance.**
