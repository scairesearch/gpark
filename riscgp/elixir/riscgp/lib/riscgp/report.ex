defmodule RiscGP.Report do
  @moduledoc """
  Markdown rendering of the study.

  Every table cell that comes from a model number is printed with its
  provenance. A report that mixes estimates and measurements without saying
  which is which is how a study turns into a press release.
  """

  alias RiscGP.Model.{Params, Study, Workload}

  @doc "The full study report as markdown."
  @spec to_markdown(Study.t(), keyword()) :: String.t()
  def to_markdown(study, opts \\ []) do
    sweep = Keyword.get(opts, :sweep, :edge)
    sections = [
      header(sweep),
      provenance_section(study.params),
      results_section(study, sweep),
      comparisons_section(study, sweep),
      gate_section(study),
      sensitivity_section(study),
      kill_criteria_section(),
      caveat_section()
    ]

    Enum.join(sections, "\n\n")
  end

  defp header(:all), do: "# riscgp P0 analytic model, full sweep"

  defp header(sweep),
    do:
      "# riscgp P0 analytic model (#{sweep} configuration)\n\n" <>
        "All numbers below are `estimate` unless a row says `measured`. " <>
        "No riscgp silicon exists, so no number here is `measured`."

  defp provenance_section(params) do
    """
    ## Parameter provenance

    | statistic | value |
    |---|---|
    | parameters | #{length(params)} |
    | marked `measured` | #{Params.measured_count(params)} |
    | marked `estimate` | #{length(params) - Params.measured_count(params)} |
    | weakest provenance | `#{Params.provenance(params)}` |
    """
  end

  defp results_section(study, sweep) do
    rows =
      study.results
      |> filter_product(sweep)
      |> Enum.sort_by(&{&1.product, &1.node, &1.datapath, &1.workload_id})
      |> Enum.map(fn r ->
        "| #{r.product} | #{r.node} | #{path(r.datapath)} | #{r.workload_id} | " <>
          "#{format_seconds(r.seconds)} | #{r.bound} | #{format_energy(r.energy.total)} | " <>
          "#{r.energy.dominant_part} | #{r.sram_kb_total} | `#{r.provenance}` |"
      end)

    """
    ## Results: cycles, energy, and what is actually binding

    | product | node | path | workload | seconds | binding resource | joules/unit | dominant energy term | die SRAM kB | provenance |
    |---|---|---|---|---|---|---|---|---|---|
    #{Enum.join(rows, "\n")}
    """
  end

  defp comparisons_section(study, sweep) do
    rows =
      study.comparisons
      |> filter_product(sweep)
      |> Enum.sort_by(&{&1.product, &1.node, &1.datapath, &1.workload_id})
      |> Enum.map(fn c ->
        "| #{c.product} | #{c.node} | #{path(c.datapath)} | #{c.workload_id} | " <>
          "#{format_seconds(c.our_seconds)} | #{ratio(c.ratios.equal_throughput)} | " <>
          "#{ratio(c.ratios.equal_latency_100ms)} | #{ratio(c.ratios.equal_latency_50ms)} | " <>
          "#{format_energy(c.reference_energy_per_unit)} | `#{c.verdict}` |"
      end)

    """
    ## Comparison against the NVIDIA H100 reference

    Conventions, most conservative first:

    1. `equal_throughput`: each part runs at its own best decode rate. The only
       convention that cannot be accused of choosing a slow operating point.
    2. `equal_latency_100ms`: both parts must deliver a token within 100 ms;
       our part is charged at the reference board power for that budget.
    3. `equal_latency_50ms`: the same at a 50 ms budget.

    A ratio above 1.0 means we use less energy per unit than the reference.

    | product | node | path | workload | our seconds | equal throughput | equal latency 100ms | equal latency 50ms | reference J/unit | KC1 |
    |---|---|---|---|---|---|---|---|---|---|
    #{Enum.join(rows, "\n")}
    """
  end

  defp gate_section(study) do
    g = study.gate

    """
    ## P0 gate status (model prediction, not measurement)

    | criterion | predicted value | status |
    |---|---|---|
    | KC1 equal-throughput ratio on WL1b | #{format_ratio(g.kc1_equal_throughput_ratio)} | `#{g.kc1_status}` |
    | KC2 WL1b meets 100 ms SLA at SKY130 | #{g.kc2_meets_100ms_sla} | `#{g.kc2_status}` |
    | KC3 WL2 is a loss (negative control) | #{g.kc3_negative_control_loss} | `#{g.kc3_status}` |
    | KC4 WL1 is SLA-infeasible (negative control) | #{g.kc4_wl1_sla_infeasible} | `#{g.kc4_status}` |

    The gate is **not** closed by this table. KC1 is a measurement criterion
    (validation layer L6) and the reference above is an analytic estimate. The
    plan permits no RTL until the gate closes on measured numbers.
    """
  end

  defp sensitivity_section(%Study{sensitivity: []}), do: "## Sensitivity\n\nNot run."

  defp sensitivity_section(%Study{sensitivity: rows}) do
    table =
      rows
      |> Enum.map(fn r ->
        "| #{r.parameter} | x#{r.factor} | #{format_energy(r.energy_per_unit)} | " <>
          "#{r.dominant_energy} | #{format_ratio(r.ratios.equal_throughput)} | " <>
          "#{ratio(r.ratios.equal_latency_100ms)} | `#{r.verdict}` | #{r.flips_kc1} |"
      end)

    """
    ## Sensitivity of the S1 axis to the parameters it depends on

    Target: edge / SKY130 / Path A / WL1b. Each parameter is swept alone.

    | parameter | factor | joules/token | dominant term | equal throughput | equal latency 100ms | KC1 | flips KC1 |
    |---|---|---|---|---|---|---|---|
    #{Enum.join(table, "\n")}
    """
  end

  defp kill_criteria_section do
    rows =
      Study.kill_criteria()
      |> Enum.map(fn kc ->
        "| #{kc.id} | #{kc.track} | #{kc.statement} | #{kc.evaluated_by} |"
      end)

    """
    ## Kill criteria, pre-registered

    | id | track | criterion | evaluated by |
    |---|---|---|---|
    #{Enum.join(rows, "\n")}

    A killed criterion stays killed. It is not re-derived from a later model.
    """
  end

  defp caveat_section do
    """
    ## What this report is not

    - Not a measurement. Every number is analytic and every coefficient is an
      estimate. The plan requires a rented H100 or B200 running these exact
      workloads before the gate closes.
    - Not a vendor comparison of record. The H100 figures here are published
      specifications used to predict a measurement, which the plan forbids as a
      primary baseline.
    - Not a density or peak-FLOPS argument. An accessible node is three to four
      lithography generations behind the parts being compared, and that gap is
      not a design-effort gap.
    - Not a claim about Bolt Zeus, Tenstorrent, Cerebras, or AMD. Those are
      reference points for S2, and Bolt has to be measured directly before any
      S2 claim, per the plan.
    """
  end

  defp filter_product(rows, :all), do: rows
  defp filter_product(rows, sweep), do: Enum.filter(rows, &(&1.product == sweep))

  defp path(:path_a), do: "A (RVV 1.0 + ext)"
  defp path(:path_b), do: "B (RV32IM + coprocessor)"

  defp format_seconds(seconds), do: :erlang.float_to_binary(seconds * 1000.0, decimals: 3) <> " ms"
  defp format_energy(joules), do: :erlang.float_to_binary(joules, decimals: 6) <> " J"
  defp format_ratio(nil), do: "n/a"
  defp format_ratio(ratio), do: :erlang.float_to_binary(ratio, decimals: 2) <> "x"

  defp ratio(nil), do: "n/a (SLA not met)"
  defp ratio(ratio), do: format_ratio(ratio)

  @doc "JSON encoding of a study result set, used by the Python parity test."
  @spec to_json(Study.t()) :: String.t()
  def to_json(study) do
    payload = %{
      "workloads" => Enum.map(Workload.all(), &%{"id" => &1.id, "kind" => Atom.to_string(&1.kind)}),
      "comparisons" =>
        Enum.map(study.comparisons, fn c ->
          %{
            "product" => Atom.to_string(c.product),
            "node" => Atom.to_string(c.node),
            "datapath" => Atom.to_string(c.datapath),
            "workload_id" => c.workload_id,
            "our_seconds" => round6(c.our_seconds),
            "our_energy_per_unit" => round6(c.our_energy_per_unit),
            "reference_seconds" => round6(c.reference_seconds),
            "reference_energy_per_unit" => round6(c.reference_energy_per_unit),
            "equal_throughput_ratio" => round6(c.ratios.equal_throughput),
            "meets_sla_100ms" => c.our_meets_sla_100ms
          }
        end),
      "gate" => %{
        "kc1_equal_throughput_ratio" => round6(study.gate.kc1_equal_throughput_ratio),
        "kc1_status" => Atom.to_string(study.gate.kc1_status),
        "kc2_meets_100ms_sla" => study.gate.kc2_meets_100ms_sla,
        "kc3_negative_control_loss" => study.gate.kc3_negative_control_loss,
        "kc4_wl1_sla_infeasible" => study.gate.kc4_wl1_sla_infeasible
      }
    }

    RiscGP.Json.encode!(payload)
  end

  defp round6(nil), do: nil
  defp round6(value) when is_float(value), do: Float.round(value, 6)
  defp round6(value), do: value
end