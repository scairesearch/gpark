defmodule RiscGP.Json do
  @moduledoc """
  Minimal JSON encoder for study output.

  The study has no runtime dependencies and the P0 mirror has to be
  byte-comparable with Python, so this writes the small subset of JSON the
  reports need instead of pulling in a library.
  """

  @spec encode!(term()) :: String.t()
  def encode!(term), do: IO.iodata_to_binary(value(term))

  defp value(nil), do: "null"
  defp value(true), do: "true"
  defp value(false), do: "false"
  defp value(value) when is_integer(value), do: Integer.to_string(value)
  defp value(value) when is_atom(value), do: inspect(value)
  defp value(value) when is_binary(value), do: [?", escape(value), ?"]
  defp value(value) when is_float(value), do: :erlang.float_to_binary(value, [:short])
  defp value(value) when is_list(value), do: [?[, Enum.map_intersperse(value, ?,, &value/1), ?]]

  defp value(value) when is_map(value) do
    inner =
      value
      |> Enum.map(fn {key, item} -> [value(to_string(key)), ?:, value(item)] end)
      |> Enum.intersperse(?,)

    [?{, inner, ?}]
  end

  defp escape(binary) do
    binary
    |> String.replace("\\", "\\\\")
    |> String.replace("\"", "\\\"")
    |> String.replace("\n", "\\n")
    |> String.replace("\t", "\\t")
    |> String.replace("\r", "\\r")
  end
end