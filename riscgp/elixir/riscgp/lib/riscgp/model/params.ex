defmodule RiscGP.Model.Params do
  @moduledoc """
  The complete input set of the P0 analytic model, with provenance on every
  entry.

  Names are `<product>.<node>.<suffix>` where a value depends on product
  configuration or process node, because node is a *parameter set* here rather
  than an assumption (plan P3.20) and because all three strategy tracks are
  funded in parallel (S1 edge SoC, S2 datacenter GPU, S3 IP).
  """

  alias RiscGP.Model.Param

  @products [:edge, :dc]
  @nodes [:sky130, :cgf22]

  @doc "Product configurations the study sweeps. S3 (IP) needs no hardware sweep."
  def products, do: @products

  @doc "Node parameter sets. SKY130 for the S1 v1 shuttle, 22nm for S2 and for any S1 claim that SKY130 cannot carry."
  def nodes, do: @nodes

  @spec scoped_name(atom(), atom(), String.t()) :: String.t()
  def scoped_name(product, node, suffix), do: "#{product}.#{node}.#{suffix}"

  @spec all() :: [Param.t()]
  def all do
    shared() ++ reference_params() ++
      for(product <- @products, node <- @nodes, do: config(product, node))
  end

  defp shared do
    [
      Param.estimated("fabric.noc_flit_bytes", 8.0, "B"),
      Param.estimated("fabric.noc_hops_mean", 3.0, "hops", "mean mesh distance"),
      Param.estimated("fabric.dma_engines", 2, "engines"),
      Param.estimated("model.tile_boundary_traffic_fraction", 0.01, "ratio", "share of read bytes crossing a tile boundary"),
      Param.estimated("path_a.scalar_cores_per_tile", 1, "cores", "one RV32IM-class scalar core per tile"),
      Param.estimated("path_a.control_instructions_per_kernel", 24.0, "instr", "loop, address, predicate, fence"),
      Param.estimated("path_a.instruction_bytes", 4.0, "B"),
      Param.estimated("path_b.cores_per_tile", 5, "cores", "Tensix precedent: no FPU, no A extension"),
      Param.estimated("path_b.store_bytes_per_cycle_per_core", 0.8, "B/cyc", "6.4 bits/cycle sustained store"),
      Param.estimated("path_b.load_bytes_per_cycle_per_core", 2.3, "B/cyc", "18.3 bits/cycle sustained load"),
      Param.estimated("path_b.fpu_macs_per_instruction", 1024.0, "MAC/instr", "matrix unit instruction granularity"),
      Param.estimated("path_b.mover_bytes_per_instruction", 64.0, "B/instr", "Mover granularity"),
      Param.estimated("path_b.push_latency_cycles", 20, "cycles", "asynchronous push"),
      Param.estimated("path_b.sync_latency_cycles", 30, "cycles", "STALLWAIT/TTSync drain"),
      Param.estimated("path_b.sync_per_kernel", 1.0, "ratio", "one drain point per kernel"),
      Param.estimated("path_b.thread_roles", 3, "threads", "T0 math, T1 unpack, T2 pack")
    ]
  end

  defp reference_params do
    [
      Param.estimated("reference.nvidia.static_watts", 450.0, "W", "clock and array power whenever not in a low power state"),
      Param.estimated("reference.nvidia.idle_watts", 200.0, "W", "to be replaced by harness output"),
      Param.estimated("reference.nvidia.hbm_bytes_per_second", 3.35e12, "B/s", "H100 SXM5 published HBM3 bandwidth"),
      Param.estimated("reference.nvidia.hbm_pj_per_bit", 3.9, "pJ/bit", "HBM3 DRAM-inclusive, published class"),
      Param.estimated("reference.nvidia.bf16_flops_per_second", 989.0e12, "FLOP/s", "dense bf16, published"),
      Param.estimated("reference.nvidia.decode_latency_seconds", 0.003, "s", "dependent-kernel latency floor at batch=1")
    ]
  end

defp config(:edge, node), do: core_config(:edge, node)

defp config(:dc, node), do: core_config(:dc, node)

defp core_config(product, node) do
    opts =
      case {product, node} do
        {:edge, :sky130} -> [tiles: 16, rows: 4, cols: 4, sram_kb: 128, dram_mt_s: 8533]
        {:edge, :cgf22} -> [tiles: 16, rows: 4, cols: 4, sram_kb: 128, dram_mt_s: 9600]
        {:dc, :sky130} -> [tiles: 64, rows: 8, cols: 8, sram_kb: 256, dram_mt_s: 6400]
        {:dc, :cgf22} -> [tiles: 64, rows: 8, cols: 8, sram_kb: 256, dram_mt_s: 6400]
      end

    dram_channels = if product == :dc, do: 4, else: 2
    dram_bits = 64
    n = fn suffix -> scoped_name(product, node, suffix) end
    wide = node == :cgf22
    clock = if wide, do: 1000.0, else: 500.0
    read_bpc = if wide, do: 64.0, else: 16.0
    write_bpc = if wide, do: 32.0, else: 8.0
    read_port_bits = if wide, do: 128, else: 64
    dram_bytes_per_cycle =
      dram_channels * dram_bits / 8 * opts[:dram_mt_s] * 1_000_000 / (2 * clock * 1_000_000)

    [
      Param.estimated(n.("tiles"), opts[:tiles], "tiles"),
      Param.estimated(n.("tile_rows"), opts[:rows], "tiles"),
      Param.estimated(n.("tile_cols"), opts[:cols], "tiles"),
      Param.estimated(n.("sram_kb_per_tile"), opts[:sram_kb], "kB"),
      Param.estimated(n.("sram_banks_per_tile"), 4, "banks"),
      Param.estimated(n.("clock_mhz"), clock, "MHz"),
      Param.estimated(n.("sram_read_bytes_per_cycle"), read_bpc, "B/cyc/tile", "per tile, shared by both datapaths"),
      Param.estimated(n.("sram_write_bytes_per_cycle"), write_bpc, "B/cyc/tile", "per tile, shared by both datapaths"),
      Param.estimated(n.("sram_read_port_bits"), read_port_bits, "b"),
      Param.estimated(n.("sram_read_pj_per_access"), if(wide, do: 30.0, else: 12.0), "pJ", "#{read_port_bits}-bit read"),
      Param.estimated(n.("sram_write_pj_per_access"), if(wide, do: 45.0, else: 18.0), "pJ", "#{read_port_bits}-bit write"),
      Param.estimated(n.("dma_pj_per_byte"), if(wide, do: 0.12, else: 0.35), "pJ/B", "DMA engine + NoC"),
      Param.estimated(n.("noc_pj_per_flit"), if(wide, do: 5.0, else: 6.0), "pJ", "#{read_port_bits}-bit flit per hop"),
      Param.estimated(n.("noc_bytes_per_cycle_per_link"), if(wide, do: 16.0, else: 8.0), "B/cyc"),
      Param.estimated(n.("dma_bytes_per_cycle"), if(wide, do: 32.0, else: 16.0), "B/cyc"),
      Param.estimated(n.("dram_channels"), opts[:dram_channels], "channels"),
      Param.estimated(n.("dram_bits_per_channel"), opts[:dram_bits], "b"),
      Param.estimated(n.("dram_bytes_per_cycle_peak"), dram_bytes_per_cycle, "B/cyc", "derived from channels x MT/s x SoC clock"),
      Param.estimated(n.("dram_efficiency"), if(wide, do: 0.75, else: 0.70), "ratio"),
      Param.estimated(n.("dram_pj_per_bit"), if(wide, do: 4.0, else: 5.0), "pJ/bit", "LPDDR5X or DDR5, DRAM-inclusive"),
      Param.estimated(n.("mac_fp32_pj"), if(wide, do: 12.0, else: 25.0), "pJ"),
      Param.estimated(n.("mac_bf16_pj"), if(wide, do: 9.0, else: 20.0), "pJ"),
      Param.estimated(n.("mac_int8_pj"), if(wide, do: 0.6, else: 1.5), "pJ"),
      Param.estimated(n.("leakage_pj_per_cycle_per_tile"), if(wide, do: 96.0, else: 24.0), "pJ", "per tile per cycle"),
      Param.estimated(n.("core_instruction_pj"), if(wide, do: 5.0, else: 8.0), "pJ"),
      Param.estimated(n.("instruction_fetch_pj_per_word"), if(wide, do: 4.0, else: 6.0), "pJ", "32-bit fetch"),
      Param.estimated(n.("vector_mac_per_cycle_fp32"), if(wide, do: 32.0, else: 16.0), "MAC/cyc/tile"),
      Param.estimated(n.("vector_mac_per_cycle_bf16"), if(wide, do: 32.0, else: 16.0), "MAC/cyc/tile"),
      Param.estimated(n.("vector_mac_per_cycle_int8"), if(wide, do: 64.0, else: 32.0), "MAC/cyc/tile"),
      Param.estimated(n.("vector_bytes_per_instruction"), if(wide, do: 32.0, else: 16.0), "B"),
      Param.estimated(n.("scalar_ipc"), 1.0, "instr/cyc"),
      Param.estimated(n.("sync_latency_cycles"), if(wide, do: 6, else: 8), "cycles"),
      Param.estimated(n.("path_b_matrix_mac_per_cycle"), if(wide, do: 256.0, else: 128.0), "MAC/cyc/tile")
    ]
  end

  @spec index([Param.t()]) :: %{String.t() => Param.t()}
  def index(params \\ all()), do: Map.new(params, &{&1.name, &1})

  @spec value([Param.t()], String.t()) :: number()
  def value(params, name) do
    case Enum.find(params, &(&1.name == name)) do
      nil -> raise KeyError, "no such param: #{name}"
      %Param{value: value} -> value
    end
  end

  @spec scoped_value([Param.t()], atom(), atom(), String.t()) :: number()
  def scoped_value(params, product, node, suffix), do: value(params, scoped_name(product, node, suffix))

  @spec put([Param.t()], String.t(), number()) :: [Param.t()]
  def put(params, name, value) do
    if Enum.any?(params, &(&1.name == name)) do
      Enum.map(params, fn %Param{name: ^name} = p -> %{p | value: value} end)
    else
      params ++ [Param.estimated(name, value, "swept")]
    end
  end

  @spec sweep([Param.t()], String.t(), number()) :: [Param.t()]
  def sweep(params, prefix, factor) do
    Enum.map(params, fn %Param{name: name} = p ->
      if String.starts_with?(name, prefix), do: %{p | value: p.value * factor}, else: p
    end)
  end

  @spec provenance([Param.t()]) :: :estimate | :measured
  def provenance(params), do: Param.weakest(params)

  @spec measured_count([Param.t()]) :: non_neg_integer()
  def measured_count(params), do: Enum.count(params, &(&1.provenance == :measured))
end