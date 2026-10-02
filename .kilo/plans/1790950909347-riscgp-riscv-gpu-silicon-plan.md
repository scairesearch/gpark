# riscgp — RISC-V GPU + edge SoC: from decision study to MPW silicon

## Context

`ceptx` contains **gpark**: kernel IR (`Gpark.IR`), typed opcode table
(`Gpark.Ops`), PTX emitter (`Gpark.PTX`), validator (`Gpark.Validate`) in Elixir
with a Python mirror. `corpus/`, `docs/`, `kernels/`, `remote/`, `riscgp/` are
empty. No commits. gpark targets **PTX only**; Metal/AMDGCN are roadmap.

`riscgp` adds a RISC-V GPU target: ISA, cycle-accurate model, gpark backend,
RTL, and MPW silicon. Also a second chip in the Apple power class (option 1 and
2 both chosen — see below).

## Facts that constrain the design (do not re-litigate these)

| Claim under discussion | Reality |
|---|---|
| "Apple chip, it's RISC-V" | **No RISC-V Apple chip exists.** A/M/H/U-series and Secure Enclave are all ARM. M3 die-shot shows no RISC-V. Real: Apple posted RISC-V roles, and SemiAnalysis (D. Patel) reports Apple is migrating *embedded* cores (Wi-Fi/BT) to RISC-V. Any "Apple" framing is a counterfactual and must be labeled one. |
| Tenstorrent as model | **Jim Keller (Tenstorrent CEO) designed Apple A4/A5, AMD Zen, Tesla FSD.** Consumer-class RISC-V efficiency talent is concentrated here, not at Apple. |
| RISC-V GPU is unoccupied | **Bolt Graphics "Zeus" taped out (Apr 2026)**, dev kits end 2026, cards 2027. Akeana core + SMT, very wide vector units, large caches, SIMT-like. One-year schedule slip. **Oxmiq Labs** sells RISC-V GPU *IP* (Raja Koduri, ex-Intel/AMD/Apple, $35M Series A: Intel Capital, Samsung, MediaTek) + OxPython (CUDA apps unmodified) + chiplets. |
| Software ecosystem is our moat | **No.** NVIDIA invested in SiFive (Series G, $400M), is integrating NVLink Fusion into RISC-V platforms, and CUDA has shipped on RISC-V. Differentiation must be in the datapath. |
| RVV gives us freedom | **RVV 1.0 ratified 2023-08-08; ratified extensions are NEVER revised.** Any extension is purely additive and cannot change core vector semantics. Post-1.0 spec churn already caused hardware compat confusion (riscv-isa-manual#1924). Do not build on unrevised 1.0 as if stable. |
| We can beat NVIDIA on density | **No.** Accessible nodes are SKY130/90nm (MPW) or 22–28nm. H100 is 4N, Blackwell further. This is a lithography gap, not a design effort. |

**Other prior art to read before P1:** NextSilicon Arbel (64/128-core RISC-V,
runtime-reconfigurable dataflow, "10x GPU at <half power"), Etched
(transformer-only ASIC), Axelera AI, d-Matrix, Fractile, SpacemiT, Eswin,
SiFive X280 Gen 2 / XM (matrix). Tenstorrent Blackhole = 120 Tensix + 16 SiFive
x280, 180 MB SRAM, 512 GB/s GDDR6, 664 TFLOPS BF8, 300 W.

## Tenstorrent, precisely (our closest design precedent)

From `tt-isa-documentation`, Wormhole/Blackhole:

- Each Tensix tile = **five RV32IM "baby" cores**, 32-bit in-order single-issue,
  *"optimized for area and power efficiency rather than for high performance."*
- Cores have **no FPU, no `A` extension**. L1 stores ~6.4 bits/cycle; loads
  ~18.3 bits/cycle sustained. Docs explicitly say never to do math on the cores.
- Cores **push instructions** to a custom coprocessor via `sw` to a magic
  address (`.ttinsn` pushes an immediate). Tensix ISA is **entirely custom**,
  *"extremely light on control flow."*
- Coprocessor backend: Matrix Unit (FPU), Unpackers, Packers, SFPU, Mover, Sync,
  plus **MOP expanders** that expand one instruction into many.
- 3 threads (T0/T1/T2) share one Matrix Unit → software pattern is one thread on
  math, one on unpack, one on pack.
- Execution is **asynchronous and not automatic**: a pushed instruction may run
  long after the core proceeds. Explicit `STALLWAIT`/TTSync required.

Two consequences: (a) **"RISC-V GPU" need not mean RISC-V does the FLOPs** — the
tiny core is where the efficiency comes from; (b) async + no coherence + explicit
fences is exactly the design class that is *expensive to verify*.

## Strategy funnel — 3 paths, picked in P0

The user wants all three kept open. They are **not** independent choices: each
implies a chip portfolio and an ISA datapath, so this is a 3-way decision with 2
derivations, not a 12-way search.

| Path | Posture | Implied chips | Implied ISA datapath | Capex |
|---|---|---|---|---|
| **S1 Differentiate** (recommended) | Own one narrow axis Bolt/others avoid. Bolt is going after path tracing, HPC, huge-memory. Not low-latency decode, not fixed-model efficiency. | Edge/consumer SoC primary; GPU as the same tile family | Coprocessor (Path B) — maximal fusion, needed to win on a narrow axis | MPW only |
| **S2 Head-on** | General RISC-V GPU vs Bolt. Most comparable, least likely — we are a new program vs taped-out silicon with a year head start and funding. | Datacenter GPU | RVV+ext (Path A) — generality demands portability | MPW + board |
| **S3 License IP** | Sell datapath/IP like Oxmiq. | None | Datapath-clean, ISA-agnostic | Lowest |

**Pre-registered rule:** S3 is incompatible with the tapeout goal the user chose.
If S3 wins, the program converts to IP and **says so** rather than quietly
reinstating a chip. S2 requires Bolt in the measured baseline, not just NVIDIA.

## Two ISA datapath paths — built as ONE parameterized Chisel core

Swapping the datapath must not change NoC, SRAM, tile boundary, or the gpark IR.
Otherwise the study is not a fair comparison.

- **Path A — RVV 1.0 + matrix/DMA/collective extension.** Vector cores do the
  math. Portable; gpark emits a real documented ISA. Cost: additive-only
  constraint above caps fusion, and RVV's `vl`/`vtype` state model fights a
  pipelined async datapath.
- **Path B — Tenstorrent model.** Tiny RV32IM cores move data and issue commands;
  a custom matrix/vector coprocessor does all math, with expanders. Smallest
  core. Math ISA unconstrained. Cost: not portable; weak "RISC-V GPU" claim;
  custom verification with no ecosystem.

**Selection is a P0/P1 output, never an assumption.**

## Chip portfolio

- **Chip 1 — edge/consumer RISC-V SoC** (option 1). Power-class-comparable to
  Apple. Makes the 100x-vs-NVIDIA claim real, because efficiency is the axis
  where architectural simplicity beats node disadvantage. Different team and a
  different power envelope from Chip 2.
- **Chip 2 — datacenter RISC-V GPU** (option 2). Primary comparison set:
  NVIDIA (rented, measured), Cerebras WSE, AMD MI300, **Tenstorrent Wormhole /
  Blackhole**, **Bolt Zeus**. Framing as "the Apple of RISC-V" is permitted
  **only** if labeled a counterfactual in every artifact.

## The 100x claim

Never a peak-FLOPS or density claim (see node table above). P0 picks at most one
axis, with a pre-registered kill criterion. The most likely survivor is
**energy per token at batch=1** — no HBM (SRAM-resident matches decoder weight
reuse), no coherence, no tensor cores idle at low batch, tiny die amortizes the
node.

**Primary deliverable of P0 is `riscgp/docs/claims.md`:** every claim, the
measurement that tests it, the baseline, and the result *including failures*.
No claim appears anywhere external that is not in that register with a passing
result. If all axes die, the program reports that and stops.

## Ordered tasks

### P0 — Decision study (no RTL, no compiler changes). Gates everything.

1. Fix workloads precisely: one named 7B-class model at batch=1 decode, one named
   fused memory-bound kernel set, one named training-shaped matmul. Same three
   for every candidate.
2. Pick the strategy path (S1/S2/S3) using the table. Record the decision and the
   reason. If S3, stop the silicon program here.
3. Build an analytic + cycle-counting model of both datapath paths. Model op
   issue cost, SRAM port bandwidth, NoC bandwidth, **and energy**. Mark every
   number `estimate` or `measured`; state energy assumptions explicitly — the
   energy model is the one that decides the claim and the one most likely wrong.
4. Rent one H100 or B200 and run the identical workloads. **Published vendor
   numbers are not acceptable as the primary baseline** — nobody publishes
   unfavorable ones. Record precision, batch, clock state, exact kernel.
5. If S2, add a Bolt Zeus measurement to the baseline set.
6. Add Tenstorrent, Cerebras, AMD as *reference* points only — different
   architectures, never the primary claim.
7. Write `riscgp/docs/claims.md`; apply kill criteria. **Gate: if no axis
   survives, stop and report. Do not proceed to RTL.**

### P1 — ISA spec and cycle-accurate model

8. Specify before emitting: encoding, operand semantics, memory model, **exact
   latency/throughput table**, sync rules. The timing table is the real product —
   the compiler is written against it. Mirror the gpark doc structure
   (`PTX-SUBSET.md` equivalent).
9. Verilator cycle-accurate sim of the parameterized core: SRAM, NoC mesh, DMA.
   Python harness, per the repo's existing two-implementation convention.
10. Write the cycle model **from the spec, before RTL**, and keep it
    independent. Divergence between model and RTL is the bug-finding mechanism.
11. **Gate: choose Path A or Path B here.** Do not freeze the ISA before the P0
    numbers report.

### P2 — gpark backend

12. Extend `Gpark.Ops` with new opcodes + typed entries (`parts`, `dtype`,
    `ndest`, `nops`, `otypes`, `spaces`, `modifiers`, `sync`). `Gpark.IR` and
    `Gpark.PTX` both read this table; the new backend must too, so validation
    and emission cannot drift.
13. Add `Gpark.RiscGP` emitter mirroring `Gpark.PTX`'s structure. Instruction
    selection targets the P1 timing table, not taste.
14. Extend `Gpark.Validate` for new address spaces **and cross-tile sync** — the
    first genuinely new validation burden, currently unaccounted for.
15. Ship the Python mirror **in the same commit** as each Elixir change.
16. Populate `corpus/specs` + `corpus/golden` with byte-exact goldens, as with
    PTX. Also emit PTX for the same kernel so both backends diff on one IR.
17. **On Path B the IR must model coprocessor-vs-RISC-V instruction provenance,
    or the emitter will lie about where instructions execute.**

### P3 — RTL and verification

18. Chisel for core, NoC, tile. Reuse Chipyard/Rocket or an Akeana/SiFive core
    rather than writing one.
19. **Verification is the cost center, not the ISA.** Budget explicitly: UVM or
    property scoreboards, directed corner cases, formal on sync logic. The most
    common reason GPU projects ship late.
20. **Node is a gate, not a v1 commitment.** Start **SkyWater SKY130 via the
    Cadence/SkyWater MPW**: $10k–12k, no NDA, open PDK, Chipyard already ran
    STAC to SKY130 tapeout, 3.588 × 5.188 mm die box. If P0/P1 show SKY130 cannot
    support a surviving claim, **the same Chisel RTL retargets** to 22–28nm
    (GlobalFoundries, TSMC CyberShuttle, or ChipFoundry chipIgnite). Do not
    re-architect when the node changes.
21. DFT: scan, MBIST per SRAM bank, JTAG. Required for bring-up, routinely
    underestimated. RISC-V gives us a GDB debug path for free — a genuine Path B
    advantage, and the P0 study should price it.
22. Do not under-budget open-source signoff. Chipyard's own docs say the open
    DRC/LVS flow is *"not stable or guaranteed to produce useful results."*
    Plan for KLayout + Magic + NetGen triage on real silicon debug; consider one
    commercial signoff pass.

### P4 — Tapeout and bring-up

23. **Fix the shuttle date in P0, not P3.** SkyWater 2026 cutoffs are mid-month;
    ChipFoundry publishes fixed dates. A missed GDS cutoff costs a full cycle.
24. Foundry requirements that are not afterthoughts: bounding box, seal ring,
    die ID. Failing these rejects the submission.
25. Bring-up order: FPGA single tile → 2×2 → silicon. Debug via JTAG/GDB.
26. Re-run every validation layer on real silicon. Model-vs-silicon divergence
    is expected and is data, not failure.

## Validation layers

Each gates the next. L0–L2 before any RTL. L4–L5 gate tapeout submission.

| Layer | Check | Gate |
|---|---|---|
| **L0** | `Gpark.Validate.check_all` on every corpus kernel; Python mirror byte-identical | before emission |
| **L1** | Verilator result == NumPy reference, bit-exact, all three workloads | before RTL |
| **L2** | Spec-derived cycle model == Verilator cycles per kernel. **Divergence is the bug finder** | before RTL |
| **L3** | ISA compliance: `riscv-tests` style for RV/RVV subset; custom suite for the extension incl. async and fence semantics | before P&R |
| **L4** | FPGA == model, cycle count and result | before tapeout |
| **L5** | Silicon == model; DRAM/NoC/SRAM margins measured against P1 timing table | bring-up |
| **L6** | Comparative: same workload on our part **and** rented NVIDIA, same precision/batch (+ Bolt if S2). Populates `claims.md` | the 100x claim |

L2 is the highest-value cheap check and the one most likely to be skipped. Don't
skip it.

## Risks

1. **The 100x claim dies in P0.** Most likely failure. Plan survives by
   reporting, not re-targeting.
2. **Two chips doubles the program.** Chip 1 and Chip 2 have different power
   envelopes, teams, and verification. Sequence explicitly: only one is in
   active build at a time, and P0 says which comes first.
3. **Verification consumes the team.** Async datapath, no coherence, explicit
   fences — the expensive class.
4. **SKY130 too old for any surviving claim.** Mitigated by treating node as a
   retarget (P3.20).
5. **Open-source signoff instability.** Documented by Chipyard itself.
6. **"All paths open" becomes a hedge.** Controlled by the pre-registered rule
   that P0 commits to one strategy and one ISA path, and that a killed option
   stays killed.
7. **Path B weakens the "RISC-V GPU" claim.** A no-FPU core doing no arithmetic
   is hard to defend as a RISC-V *GPU*. Decide honest wording before any
   external communication.
8. **Bolt ships first.** Dev kits end 2026. By the time we have silicon they
   have a product; the differentiation must be pre-decided in P0, not found
   later.
9. **Shuttle calendar slip.** Fix the date in P0.

## Open questions

1. **Which strategy path (S1/S2/S3)?** Blocks everything. Output of P0 step 2.
2. **Which axis survives P0?** Blocks the datapath. If none, stop.
3. **Path A or Path B?** Blocks the ISA freeze at P1.11.
4. **Which chip is built first, and are both funded at all?** Two chips is two
   programs; the plan sequences but does not assume both proceed.
5. **Node for v1: SKY130 or 22–28nm?** Decided at the P1/P2 boundary.
6. **Is `riscgp` a gpark backend or a standalone compiler?** Recommend
   backend-first: `Gpark.IR`, the typed opcode table, and the corpus/golden
   harness already exist for exactly this.

---

# Session decision record

Complete record of this planning session: what was asked, what was answered, and
what I corrected. Kept because several answers were ambiguous and the resolution
is not recoverable from the plan text alone. This is a summary, not a verbatim
transcript.

## Initial request

> "explore plan for RISV GPU 100x better than nvidia and cerebras and amd"
> "physical ISA, and more"
> "new gpu hardwar and apple chip, its riscv, use tensetorrent and other variets"
> "all add Tenstorrent to exploration"
> "we were exploring new session dont look at other folders"

Scope constraints established: **stay inside `riscgp`** — do not read other
folders in the workspace. Tenstorrent added to the prior-art set.

## Repo state as found

`ceptx` is a skeleton with **no commits** (`git log` → "branch 'main' does not
have any commits yet"). gpark = `Gpark.IR` / `Gpark.Ops` / `Gpark.PTX` /
`Gpark.Validate` in Elixir, Python mirror intended. `corpus/specs`,
`corpus/golden`, `docs/`, `kernels/`, `remote/`, `riscgp/` all **empty**. The
CHANGELOG references `docs/CONTEXT.md`, `ROADMAP.md`, `ARCHITECTURE.md`,
`PTX-SUBSET.md`, `VALIDATION.md` etc. — **`docs/` is empty, so those files do
not exist** despite the changelog. Flagged as context for the implementer: do
not go looking for the roadmap the changelog promises.

Relevant source detail an implementer needs: `Gpark.IR.op_spec/1` is defined
**twice** (line 121 `defdelegate` → `Gpark.Ops.fetch/1`, then line 126
`def ops/0` → `Map.get(@ops, base)`), and the second set references `@ops`,
which does not exist in `Gpark.IR`. Dead or broken code — resolve before
extending the opcode table, because `Gpark.Ops` is the table P2 extends.

## Q&A log

| # | Question | Answer given | Resolution in plan |
|---|---|---|---|
| 1 | What does "100x better" mean — which axis? | *"we were exploring new session dont look at other folders"* (axis left open; Tenstorrent added) | Kept open deliberately, converted into P0 with kill criteria |
| 2 | What does "physical ISA" mean? | **"Full custom silicon to tapeout"** | P3/P4; MPW-first, node as a retarget not a v1 commitment |
| 3 | How should the ISA relate to RVV 1.0? | *"all add Tenstorrent to exploration"* | Left open; resolved as Path A vs Path B, selected in P0/P1 |
| 4 | Which 100x axis? | **"Explore all three before committing"** | Structure re-cut as a *cheap gated study*, not a hedge — pre-registered kill criteria, all three modeled, P0 commits to ≤1 |
| 5 | What is v1? | **"all"** | Same: staged tranches, each gated on P0 numbers |
| 6 | RISC-V cores vs separate coprocessor? | **"Build both paths, let the study pick"** | Path A / Path B as **one parameterized Chisel core** so the study compares real datapaths |
| 7 | How is 100x measured? | **"all"** | Rented NVIDIA is the primary baseline; Tenstorrent/Cerebras/AMD as reference only |
| 8 | What does "Apple chip, it's RISC-V" mean? | **"1 and 2"** | Both: Chip 1 = real edge/consumer SoC; Chip 2 = datacenter GPU, with the "Apple of RISC-V" framing permitted **only** when labelled a counterfactual |
| 9 | How to position vs Bolt and Oxmiq? | **"add all 3 into different paths so we can choose"** | S1/S2/S3 strategy funnel, committed in P0 step 2. **Combinatorics resolved**: 2 chips × 2 datapaths × 3 strategies = 12, but each strategy implies a chip portfolio and a datapath → 3-way decision with 2 derivations each |

## Corrections made (these changed the plan)

1. **"100x better than NVIDIA" is not a spec.** Re-framed as a named axis with a
   pre-registered kill criterion. On accessible nodes (SKY130/90nm MPW, 22–28nm)
   a peak-FLOPS or density claim is a lithography gap, not a design effort.
2. **"Apple chip, it's RISC-V" is a counterfactual.** No RISC-V Apple chip
   exists; A/M/H/U-series and the Secure Enclave are ARM. M3 die-shot shows no
   RISC-V cores. Real: Apple posted RISC-V roles; SemiAnalysis (D. Patel) reports
   embedded (Wi-Fi/BT) migration.
3. **The RISC-V GPU space is not empty.** Bolt Graphics "Zeus" taped out
   (Apr 2026), dev kits end 2026, cards 2027. Oxmiq Labs sells RISC-V GPU IP,
   led by Raja Koduri (ex-Intel/AMD/Apple), $35M Series A. Both added to the
   plan as baseline/competitor.
4. **"Software ecosystem" is not a moat.** NVIDIA invested in SiFive (Series G
   $400M), is integrating NVLink Fusion into RISC-V platforms, and CUDA has
   shipped on RISC-V.
5. **RVV 1.0 is not a freedom.** Ratified 2023-08-08; ratified extensions are
   **never revised**, so any extension is purely additive. Caps fusion on
   Path A and is the main argument for Path B.

## Research findings the plan depends on

- **MPW cost makes tapeout accessible**: ≥180nm $3k–30k; 130–55nm $8k–80k;
  40–28nm $25k–200k; ≤22nm $50k–400k. **SkyWater SKY130 via Cadence MPW is
  $10k–12k, no NDA, open PDK**, die box 3.588 × 5.188 mm, 2026 cutoffs
  mid-month. Chipyard already ran STAC to SKY130 tapeout.
  Leading-edge for contrast: 3nm NRE $400–600M+, masks $15M+.
- **Tenstorrent** (`tt-isa-documentation`): 5× RV32IM in-order single-issue baby
  cores per Tensix, no FPU, no `A`; L1 stores ~6.4 bits/cycle, loads ~18.3
  bits/cycle; cores `sw`-push instructions to a custom coprocessor (`.ttinsn`
  pushes an immediate); coprocessor = Matrix Unit, Unpackers, Packers, SFPU,
  Mover, Sync + **MOP expanders**; 3 threads share one Matrix Unit; execution
  asynchronous with explicit `STALLWAIT`/TTSync. Blackhole: 120 Tensix + 16
  SiFive x280, 180 MB SRAM, 512 GB/s GDDR6, 664 TFLOPS BLOCKFP8, 300 W.
  Wormhole FlashAttention: 20x over their own baseline via async + pipelining.
  CEO **Jim Keller designed Apple A4/A5, AMD Zen, Tesla FSD**.
- **RISC-V spec state**: RVV 1.0 ratified 2023-08-08; 1.1 still draft;
  post-ratification churn caused hardware compat confusion
  (riscv-isa-manual#1924).
- **Chipyard's own caveat**: the open-source DRC/LVS signoff flow is *"not
  stable or guaranteed to produce useful results."* Basis for P3.22.
- **Open-source ASIC flow exists and works**: Chipyard + Hammer + Sky130 tech
  plugin + OpenROAD (Yosys/OpenROAD/KLayout/Magic/NetGen), sram22 SRAM macros.

## Standing instruction

P0 is a **kill-oriented** study, not a confirmation exercise. If no 100x axis
survives, or if S3 wins, the program reports that and stops rather than
reinstating a chip target. Any option killed in P0 stays killed.
