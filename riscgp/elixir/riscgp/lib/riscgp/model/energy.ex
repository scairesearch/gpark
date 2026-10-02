defmodule RiscGP.Model.Energy do
  @moduledoc """
  Joules per workload invocation, decomposed by source.

  This is the part of the model most likely to be wrong, and it is the part
  that decides the claim (plan P0.3). Every coefficient carries provenance and
  the report prints the provenance with the result.

  Formulas are stated in riscgp/docs/MODEL.md.
  """

  alias RiscGP.Model.{Fabric, Kernel, Params, Schedule, Workload}

  defstruct [
    :total,
    :parts,
    :dominant_part,
    :energy_per_unit,
    :provenance,
    :watts
  ]

  @type t :: %__MODULE__{}

  @spec run(Workload.t(), Schedule.t(), [Params.t()], atom(), atom()) :: t()
  def run(%{} = workload, schedule, params, product, node) do
    read_bytes = Kernel.total_read_bytes(workload.kernels)
    write_bytes = Kernel.total_write_bytes(workload.kernels)
    tiles = Fabric.tiles(params, product, node)
    boundary_bytes = Fabric.tile_boundary_bytes(params, workload)

    parts = %{
      mac: mac_pj(workload, params, product, node),
      sram_read: sram_pj(read_bytes, "sram_read_bytes_per_cycle", "sram_read_pj_per_access", params, product, node),
      sram_write: sram_pj(write_bytes, "sram_write_bytes_per_cycle", "sram_write_pj_per_access", params, product, node),
      noc: noc_pj(boundary_bytes, params, product, node),
      dma: (read_bytes + write_bytes) * Fabric.scoped(params, product, node, "dma_pj_per_byte"),
      dram: (read_bytes + write_bytes) * 8 * Fabric.scoped(params, product, node, "dram_pj_per_bit"),
      core: core_pj(schedule, params, product, node),
      leakage: tiles * schedule.cycles * Fabric.scoped(params, product, node, "leakage_pj_per_cycle_per_tile")
    }

    total = parts |> Map.values() |> Enum.sum()

    %__MODULE__{
      total: total / 1.0e12,
      parts: Map.new(parts, fn {key, pj} -> {key, pj / 1.0e12} end),
      dominant_part: Enum.max_by(parts, fn {_key, pj} -> pj end) |> elem(0),
      energy_per_unit: total / 1.0e12,
      provenance: Params.provenance(params),
      watts: total / 1.0e12 / max(schedule.seconds, 1.0e-12)
    }
  end

  defp mac_pj(workload, params, product, node) do
    Enum.reduce(workload.kernels, 0.0, fn kernel, acc ->
      suffix =
        case kernel.mac_kind do
          :int8 -> "mac_int8_pj"
          :bf16 -> "mac_bf16_pj"
          :fp32 -> "mac_fp32_pj"
        end

      acc + kernel.macs * Fabric.scoped(params, product, node, suffix)
    end)
  end

  defp sram_pj(bytes, width_suffix, energy_suffix, params, product, node) do
    bytes_per_access = Fabric.scoped(params, product, node, width_suffix)
    bytes / bytes_per_access * Fabric.scoped(params, product, node, energy_suffix)
  end

  defp noc_pj(boundary_bytes, params, product, node) do
    flit_bytes = Params.value(params, "fabric.noc_flit_bytes")
    hops = Params.value(params, "fabric.noc_hops_mean")
    flit_pj = Fabric.scoped(params, product, node, "noc_pj_per_flit")
    boundary_bytes / flit_bytes * hops * flit_pj
  end

  defp core_pj(schedule, params, product, node) do
    instruction_bytes = Params.value(params, "path_a.instruction_bytes")
    instruction_pj = Fabric.scoped(params, product, node, "core_instruction_pj")
    fetch_pj = Fabric.scoped(params, product, node, "instruction_fetch_pj_per_word")

    schedule.instructions * instruction_pj +
      schedule.instructions * instruction_bytes / 4 * fetch_pj
  end
end