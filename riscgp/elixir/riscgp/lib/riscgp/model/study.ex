defmodule RiscGP.Model.Study do
  @moduledoc """
  Runs the P0 model over products, nodes, datapaths, and the fixed workloads,
  then compares each result against the NVIDIA reference under three
  pre-registered conventions.

  The conventions matter more than the numbers: a ratio without a named
  comparison convention is unfalsifiable, and the convention is what decides
  whether a 100x claim is a result or a rounding choice.
  """

  alias RiscGP.Model.{Energy, Fabric, Kernel, Params, Schedule, Workload}

  defstruct [:results, :comparisons, :sensitivity, :gate, :params]

  @type t :: %__MODULE__{}

  @conventions [:equal_throughput, :equal_latency_100ms, :equal_latency_50ms]

  @doc "The three pre-registered comparison conventions, most conservative first."
  def conventions, do: @conventions

  @doc "Pre-registered kill criteria. Applied to the model now, to the measurement later."
  def kill_criteria do
    [
      %{
        id: "KC1",
        track: "S1",
        statement:
          "WL1b energy per token must beat the measured NVIDIA baseline by at least 10x under " <>
            "the most conservative convention (equal throughput).",
        verdict_when: :fail,
        evaluated_by: "L6 measurement, predicted here by the analytic model"
      },
      %{
        id: "KC2",
        track: "S1",
        statement: "WL1b must meet a 100 ms/token latency SLA on the SKY130 edge configuration.",
        verdict_when: :fail,
        evaluated_by: "analytic model now, silicon at L5"
      },
      %{
        id: "KC3",
        track: "S1",
        statement:
          "WL2, the pre-registered negative control, must be a loss. A win on WL2 invalidates " <>
            "the model rather than the design.",
        verdict_when: :fail_if_pass,
        evaluated_by: "analytic model now, L6 measurement later"
      },
      %{
        id: "KC4",
        track: "S1",
        statement:
          "WL1 must be reported as SLA-infeasible on the SKY130 edge configuration. If WL1 looks " <>
            "feasible the memory model is wrong.",
        verdict_when: :fail_if_pass,
        evaluated_by: "analytic model"
      }
    ]
  end

  @doc "Runs the full sweep."
  @spec run(keyword()) :: t()
  def run(opts \\ []) do
    params = Keyword.get(opts, :params, Params.all())
    products = Keyword.get(opts, :products, Params.products())
    nodes = Keyword.get(opts, :nodes, Params.nodes())
    workloads = Keyword.get(opts, :workloads, Workload.all())

    results =
      for product <- products,
          node <- nodes,
          datapath <- Schedule.datapaths(),
          workload <- workloads do
        result(product, node, datapath, workload, params)
      end

    comparisons =
      for product <- products,
          node <- nodes,
          datapath <- Schedule.datapaths(),
          workload <- workloads do
        comparison(result(product, node, datapath, workload, params), params)
      end

    %__MODULE__{
      params: params,
      results: results,
      comparisons: comparisons,
      sensitivity: Keyword.get(opts, :sensitivity, []),
      gate: gate(comparisons)
    }
  end

  @doc "One product, node, datapath, workload point of the sweep."
  @spec result(atom(), atom(), Schedule.datapath(), Workload.t(), [Params.t()]) :: map()
  def result(product, node, datapath, workload, params \\ Params.all()) do
    schedule = Schedule.run(workload, params, product, node, datapath)
    energy = Energy.run(workload, schedule, params, product, node)

    %{
      product: product,
      node: node,
      datapath: datapath,
      workload_id: workload.id,
      workload_name: workload.name,
      kind: workload.kind,
      units: workload.units,
      seconds: schedule.seconds,
      cycles: schedule.cycles,
      bound: schedule.bound,
      bounds: schedule.bounds,
      energy: energy,
      tokens_per_second: units_per_second(workload, schedule.seconds),
      sram_kb_total: Fabric.sram_kb_total(params, product, node),
      provenance: Params.provenance(params),
      meets_sla_100ms: schedule.seconds <= 0.1
    }
  end

  defp units_per_second(%{kind: :decode}, seconds), do: 1.0 / seconds
  defp units_per_second(_workload, seconds), do: 1.0 / seconds

  @doc "Compares one of our points against the NVIDIA reference under every convention."
  @spec comparison(map(), [Params.t()]) :: map()
  def comparison(result, params \\ Params.all()) do
    workload = Workload.find(result.workload_id)
    reference = nvidia_reference(workload, params)

    ratios =
      Map.new(@conventions, fn convention ->
        {convention, convention_ratio(convention, result, reference)}
      end)

    %{
      product: result.product,
      node: result.node,
      datapath: result.datapath,
      workload_id: result.workload_id,
      kind: result.kind,
      our_seconds: result.seconds,
      our_energy_per_unit: result.energy.total,
      our_tokens_per_second: result.tokens_per_second,
      our_watts: result.energy.watts,
      our_dominant_energy: result.energy.dominant_part,
      our_bound: result.bound,
      our_meets_sla_100ms: result.meets_sla_100ms,
      reference_seconds: reference.seconds,
      reference_energy_per_unit: reference.energy,
      ratios: ratios,
      verdict: verdict(ratios)
    }
  end

  defp convention_ratio(:equal_throughput, result, reference) do
    reference.energy / result.energy.total
  end

  defp convention_ratio(convention, result, reference) when convention in [:equal_latency_100ms, :equal_latency_50ms] do
    latency = latency_budget(convention)
    if result.seconds <= latency do
      reference.board_watts * latency / result.energy.total
    else
      nil
    end
  end

  defp latency_budget(:equal_latency_100ms), do: 0.1
  defp latency_budget(:equal_latency_50ms), do: 0.05

  defp verdict(ratios) do
    case ratios do
      %{equal_throughput: ratio} when is_number(ratio) and ratio >= 10.0 -> :survives
      _ -> :fails
    end
  end

  @doc """
  The NVIDIA reference. Never the primary baseline: the plan forbids published
  vendor numbers as the primary comparison. These numbers exist to predict what
  the harness will measure and to expose the assumptions in that prediction.
  """
  @spec nvidia_reference(Workload.t(), [Params.t()]) :: map()
  def nvidia_reference(workload, params \\ Params.all()) do
    static_watts = Params.value(params, "reference.nvidia.static_watts")
    idle_watts = Params.value(params, "reference.nvidia.idle_watts")
    board_watts = static_watts + idle_watts
    bandwidth = Params.value(params, "reference.nvidia.hbm_bytes_per_second")
    flops = Params.value(params, "reference.nvidia.bf16_flops_per_second")
    pj_per_bit = Params.value(params, "reference.nvidia.hbm_pj_per_bit")

    read_bytes = Kernel.total_read_bytes(workload.kernels)
    macs = Kernel.total_macs(workload.kernels)

    memory_seconds = read_bytes / bandwidth
    compute_seconds = 2 * macs / flops
    latency_seconds = Params.value(params, "reference.nvidia.decode_latency_seconds")

    seconds =
      case workload.kind do
        :decode -> max(memory_seconds, latency_seconds)
        _ -> max(memory_seconds, compute_seconds)
      end

    dram_energy = read_bytes * 8 * pj_per_bit

    %{
      name: "NVIDIA H100 SXM5 reference (analytic estimate, not the primary baseline)",
      seconds: seconds,
      board_watts: board_watts,
      watts: board_watts + dram_energy / max(seconds, 1.0e-12),
      energy: board_watts * seconds + dram_energy,
      tokens_per_second: 1.0 / seconds,
      provenance: :estimate
    }
  end

  @doc """
  One-at-a-time sensitivity on the parameters the verdict depends on.

  The plan calls the energy model the part most likely to be wrong. If a claim
  only survives inside the uncertainty band, it has not survived.
  """
  @spec sensitivity(keyword()) :: [map()]
  def sensitivity(opts \\ []) do
    params = Keyword.get(opts, :params, Params.all())
    factors = Keyword.get(opts, :factors, [0.5, 2.0])
    prefixes = Keyword.get(opts, :prefixes, default_sweeps())
    target = Keyword.get(opts, :target, %{product: :edge, node: :sky130, datapath: :path_a, workload_id: "wl1b_qwen3_06b_decode"})

    baseline = comparison(comparison_result(params, target), params)

    for prefix <- prefixes,
        factor <- factors do
      swept = Params.sweep(params, prefix, factor)
      result = result(target.product, target.node, target.datapath, Workload.find(target.workload_id), swept)
      comparison = comparison(result, swept)

      %{
        parameter: prefix,
        factor: factor,
        energy_per_unit: result.energy.total,
        dominant_energy: result.energy.dominant_part,
        seconds: result.seconds,
        ratios: comparison.ratios,
        verdict: comparison.verdict,
        flips_kc1: comparison.verdict != baseline.verdict
      }
    end
  end

  defp comparison_result(params, target) do
    result(target.product, target.node, target.datapath, Workload.find(target.workload_id), params)
  end

  defp default_sweeps do
    [
      "edge.sky130.sram_read_pj_per_access",
      "edge.sky130.sram_write_pj_per_access",
      "edge.sky130.dma_pj_per_byte",
      "edge.sky130.dram_pj_per_bit",
      "edge.sky130.leakage_pj_per_cycle_per_tile",
      "edge.sky130.mac_int8_pj",
      "edge.sky130.clock_mhz",
      "edge.sky130.dram_bytes_per_cycle_peak",
      "reference.nvidia.static_watts",
      "reference.nvidia.idle_watts"
    ]
  end

  @doc "Gate verdict over the sweep: which kill criteria fire on the model."
  @spec gate([map()]) :: map()
  def gate(comparisons) do
    edge = fn workload_id -> Enum.filter(comparisons, &(&1.product == :edge and &1.node == :sky130 and &1.datapath == :path_a and &1.workload_id == workload_id)) end

    wl1b = edge.("wl1b_qwen3_06b_decode")
    wl1 = edge.("wl1_llama31_8b_decode")
    wl2 = edge.("wl2_fused_int8_memory_bound")

    kc1 = wl1b |> Enum.map(& &1.ratios.equal_throughput) |> Enum.min()
    kc2 = wl1b |> Enum.map(& &1.our_meets_sla_100ms) |> Enum.all?()
    kc3 = wl2 |> Enum.map(& &1.ratios.equal_throughput) |> Enum.max()
    kc4 = wl1 |> Enum.map(& &1.our_meets_sla_100ms) |> Enum.any?()

    %{
      kc1_equal_throughput_ratio: kc1,
      kc1_status: if(kc1 >= 10.0, do: :pass, else: :fail),
      kc2_meets_100ms_sla: kc2,
      kc2_status: if(kc2, do: :pass, else: :fail),
      kc3_negative_control_loss: kc3 < 1.0,
      kc3_status: if(kc3 < 1.0, do: :pass, else: :fail),
      kc4_wl1_sla_infeasible: not kc4,
      kc4_status: if(not kc4, do: :pass, else: :fail),
      prediction_only: true
    }
  end
end