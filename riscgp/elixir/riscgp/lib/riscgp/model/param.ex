defmodule RiscGP.Model.Param do
  @moduledoc """
  One model input, carrying its provenance.

  Every number that reaches a report passes through this struct so that the
  report can print `estimate` or `measured` next to it. A parameter with no
  provenance is a bug, not a default.
  """

  @provenance [:estimate, :measured]

  defstruct [:name, :value, :unit, :provenance, :note]

  @type t :: %__MODULE__{
          name: String.t(),
          value: number(),
          unit: String.t(),
          provenance: :estimate | :measured,
          note: String.t()
        }

  @spec new(String.t(), number(), String.t(), :estimate | :measured, String.t()) :: t()
  def new(name, value, unit, provenance, note \\ "") do
    if provenance not in @provenance do
      raise ArgumentError,
            "param #{name}: provenance must be one of #{inspect(@provenance)}, got #{inspect(provenance)}"
    end

    if not is_number(value) do
      raise ArgumentError, "param #{name}: value must be a number, got #{inspect(value)}"
    end

    %__MODULE__{name: name, value: value, unit: unit, provenance: provenance, note: note}
  end

  @spec estimated(String.t(), number(), String.t(), String.t()) :: t()
  def estimated(name, value, unit, note \\ ""), do: new(name, value, unit, :estimate, note)

  @spec measured(String.t(), number(), String.t(), String.t()) :: t()
  def measured(name, value, unit, note \\ ""), do: new(name, value, unit, :measured, note)

  @doc "Marks the weakest provenance present in a collection of params."
  @spec weakest([t()]) :: :estimate | :measured | nil
  def weakest(params) do
    provenances = Enum.map(params, & &1.provenance)
    if Enum.all?(provenances, &(&1 == :measured)) do
      :measured
    else
      :estimate
    end
  end
end