defmodule Mix.Tasks.Riscgp.Study do
  @shortdoc "Runs the P0 analytic study and prints the report"

  @moduledoc """
  `mix riscgp.study` renders the P0 decision-study report.

      mix riscgp.study                 # edge configuration, markdown to stdout
      mix riscgp.study --sweep all     # both products and both nodes
      mix riscgp.study --json out.json # machine-readable, for the Python parity check

  This task computes models. It does not measure anything and it does not
  touch RTL. The plan forbids RTL until the P0 gate closes on measured numbers.
  """

  use Mix.Task

  alias RiscGP.Model.{Schedule, Study}
  alias RiscGP.Report

  @impl Mix.Task
  def run(args) do
    {opts, _rest} = OptionParser.parse!(args, strict: [sweep: :string, json: :string, no_sensitivity: :boolean])
    sweep = parse_sweep(opts[:sweep])

    study =
      Study.run(
        sensitivity: if(opts[:no_sensitivity], do: [], else: Study.sensitivity())
      )

    case opts[:json] do
      nil -> Mix.shell().info(Report.to_markdown(study, sweep: sweep))
      path -> File.write!(path, Report.to_json(study))
    end

    {:ok, study}
  end

  defp parse_sweep(nil), do: :edge
  defp parse_sweep("all"), do: :all

  defp parse_sweep(other) do
    unless other in ["edge", "dc"], do: Mix.raise("--sweep must be edge, dc, or all")
    String.to_existing_atom(other)
  end

  @doc false
  def datapaths, do: Schedule.datapaths()
end