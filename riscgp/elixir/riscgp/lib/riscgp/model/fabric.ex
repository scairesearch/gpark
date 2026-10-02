defmodule RiscGP.Model.Fabric do
  @moduledoc """
  Die-level fabric quantities that must be identical whichever datapath is
  chosen. The plan makes this a hard requirement: swapping Path A for Path B
  may not change NoC, SRAM, or the tile boundary, otherwise the two paths are
  not comparable.

  Nothing in this module reads a datapath parameter. That is the property the
  invariance test in `RiscGP.Model.StudyTest` checks.
  """

  alias RiscGP.Model.Params

  @spec tiles([Params.t()], atom(), atom()) :: number()
  def tiles(params, product, node), do: scoped(params, product, node, "tiles")

  @spec clock_hz([Params.t()], atom(), atom()) :: float()
  def clock_hz(params, product, node), do: scoped(params, product, node, "clock_mhz") * 1.0e6

  @spec sram_kb_total([Params.t()], atom(), atom()) :: number()
  def sram_kb_total(params, product, node) do
    tiles(params, product, node) * scoped(params, product, node, "sram_kb_per_tile")
  end

  @spec die_sram_read_bytes_per_cycle([Params.t()], atom(), atom()) :: float()
  def die_sram_read_bytes_per_cycle(params, product, node) do
    tiles(params, product, node) * scoped(params, product, node, "sram_read_bytes_per_cycle")
  end

  @spec die_sram_write_bytes_per_cycle([Params.t()], atom(), atom()) :: float()
  def die_sram_write_bytes_per_cycle(params, product, node) do
    tiles(params, product, node) * scoped(params, product, node, "sram_write_bytes_per_cycle")
  end

  @spec sram_read_port_bits([Params.t()], atom(), atom()) :: number()
  def sram_read_port_bits(params, product, node), do: scoped(params, product, node, "sram_read_port_bits")

  @spec dram_bytes_per_cycle_effective([Params.t()], atom(), atom()) :: float()
  def dram_bytes_per_cycle_effective(params, product, node) do
    scoped(params, product, node, "dram_bytes_per_cycle_peak") *
      scoped(params, product, node, "dram_efficiency")
  end

  @spec die_noc_bytes_per_cycle([Params.t()], atom(), atom()) :: float()
  def die_noc_bytes_per_cycle(params, product, node) do
    tiles(params, product, node) * scoped(params, product, node, "noc_bytes_per_cycle_per_link")
  end

  @spec tile_boundary_bytes([Params.t()], RiscGP.Model.Workload.t()) :: float()
  def tile_boundary_bytes(params, workload) do
    fraction = Params.value(params, "model.tile_boundary_traffic_fraction")
    RiscGP.Model.Kernel.total_read_bytes(workload.kernels) * fraction
  end

  @spec scoped([Params.t()], atom(), atom(), String.t()) :: number()
  def scoped(params, product, node, suffix), do: Params.scoped_value(params, product, node, suffix)
end