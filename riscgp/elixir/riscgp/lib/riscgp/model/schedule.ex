defmodule RiscGP.Model.Schedule do
  @moduledoc """
  Cycle model for both datapath paths.

  A path is chosen; the fabric is not. Both paths take the same kernel
  decomposition and the same fabric and differ only in what the issue rules
  cost. The `bound` field names the resource that set the time, because a cycle
  count without its bound is not a diagnosis.

  Formulas are stated in riscgp/docs/MODEL.md.
  """

  alias RiscGP.Model.{Fabric, Kernel, Params}

  @type datapath :: :path_a | :path_b

  @type t :: %__MODULE__{
          datapath: datapath(),
          product: atom(),
          node: atom(),
          cycles: float(),
          bound: atom(),
          bounds: %{atom() => float()},
          seconds: float(),
          instructions: float(),
          pushes: float(),
          sync_cycles: float()
        }

  defstruct [
    :datapath,
    :product,
    :node,
    :cycles,
    :bound,
    :bounds,
    :seconds,
    :instructions,
    :pushes,
    :sync_cycles
  ]

  @datapaths [:path_a, :path_b]

  def datapaths, do: @datapaths

  @spec run(Workload.t(), [Params.t()], atom(), atom(), datapath()) :: t()
  def run(%{} = workload, params, product, node, datapath) do
    kernels = workload.kernels
    macs = Kernel.total_macs(kernels)
    read_bytes = Kernel.total_read_bytes(kernels)
    write_bytes = Kernel.total_write_bytes(kernels)
    mac_kind = Kernel.dominant_mac_kind(kernels)
    tiles = Fabric.tiles(params, product, node)
    boundary_bytes = Fabric.tile_boundary_bytes(params, workload)

    shared = %{
      mac: macs / (tiles * vector_mac_per_cycle(params, product, node, mac_kind)),
      sram_read: read_bytes / Fabric.die_sram_read_bytes_per_cycle(params, product, node),
      sram_write: write_bytes / Fabric.die_sram_write_bytes_per_cycle(params, product, node),
      dram: read_bytes / Fabric.dram_bytes_per_cycle_effective(params, product, node),
      noc: boundary_bytes / Fabric.die_noc_bytes_per_cycle(params, product, node)
    }

    schedule =
      case datapath do
        :path_a -> path_a(shared, kernels, params, product, node, tiles, macs, read_bytes, mac_kind)
        :path_b -> path_b(shared, kernels, params, product, node, tiles, macs, read_bytes, write_bytes)
      end

    %__MODULE__{schedule | product: product, node: node, seconds: schedule.cycles / Fabric.clock_hz(params, product, node)}
  end

  defp path_a(shared, kernels, params, product, node, tiles, macs, read_bytes, mac_kind) do
    control_instructions = Params.value(params, "path_a.control_instructions_per_kernel") * Kernel.count(kernels)
    ipc = Fabric.scoped(params, product, node, "scalar_ipc")
    vector_bytes = Fabric.scoped(params, product, node, "vector_bytes_per_instruction")
    element_bytes = if mac_kind == :int8, do: 1, else: 4
    memory_instructions = read_bytes / vector_bytes
    math_instructions = macs / (vector_bytes / element_bytes)
    issue_cycles = max(memory_instructions, math_instructions) / tiles + control_instructions / (tiles * ipc)

    bounds = Map.put(shared, :issue, issue_cycles)
    sync_cycles = Kernel.count(kernels) * Fabric.scoped(params, product, node, "sync_latency_cycles")

    %__MODULE__{
      datapath: :path_a,
      bounds: bounds,
      bound: dominant_key(bounds),
      cycles: dominant(bounds) + sync_cycles,
      sync_cycles: sync_cycles,
      instructions: memory_instructions + math_instructions + control_instructions,
      pushes: 0
    }
  end

  defp path_b(shared, kernels, params, product, node, tiles, macs, read_bytes, write_bytes) do
    cores = Params.value(params, "path_b.cores_per_tile")
    store_bytes_per_cycle = tiles * cores * Params.value(params, "path_b.store_bytes_per_cycle_per_core")
    load_bytes_per_cycle = tiles * cores * Params.value(params, "path_b.load_bytes_per_cycle_per_core")
    instruction_bytes = Params.value(params, "path_a.instruction_bytes")
    fpu_pushes = Float.ceil(macs / Params.value(params, "path_b.fpu_macs_per_instruction"))
    mover_pushes = Float.ceil((read_bytes + write_bytes) / Params.value(params, "path_b.mover_bytes_per_instruction"))

    bounds =
      shared
      |> Map.merge(%{
        mac: macs / (tiles * Fabric.scoped(params, product, node, "path_b_matrix_mac_per_cycle")),
        push_store: (fpu_pushes + mover_pushes) * instruction_bytes / store_bytes_per_cycle,
        core_load: read_bytes / load_bytes_per_cycle
      })

    sync_cycles =
      Kernel.count(kernels) * Params.value(params, "path_b.sync_per_kernel") *
        Params.value(params, "path_b.sync_latency_cycles")

    pushes_per_core = Float.ceil((fpu_pushes + mover_pushes) / (tiles * cores))

    %__MODULE__{
      datapath: :path_b,
      bounds: bounds,
      bound: dominant_key(bounds),
      cycles: dominant(bounds) + sync_cycles + Params.value(params, "path_b.push_latency_cycles"),
      sync_cycles: sync_cycles,
      instructions: pushes_per_core * cores * tiles,
      pushes: fpu_pushes + mover_pushes
    }
  end

  defp vector_mac_per_cycle(params, product, node, mac_kind) do
    suffix =
      case mac_kind do
        :int8 -> "vector_mac_per_cycle_int8"
        :bf16 -> "vector_mac_per_cycle_bf16"
        :fp32 -> "vector_mac_per_cycle_fp32"
      end

    Fabric.scoped(params, product, node, suffix)
  end

  @doc "Largest bound value in a bound map."
  def dominant(bounds), do: bounds |> Map.values() |> Enum.max()

  @doc "Name of the bound that set the time."
  def dominant_key(bounds), do: Enum.max_by(bounds, fn {_key, value} -> value end) |> elem(0)
end