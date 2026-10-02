defmodule RiscGP.JSON do
  @moduledoc """
  A minimal JSON reader, just enough to load `isa/rvgpu-table.json`.

  riscgp has no dependencies on purpose. It is a standalone project and the
  table is the one file both the Elixir and the Python implementations read, so
  the only thing required is a reader for a well-behaved subset of JSON:
  objects, arrays, strings, numbers, booleans and null.

  It is not a general-purpose parser. It does not handle multiple documents,
  trailing content, or comments. It raises rather than guessing — a silently
  mis-parsed timing table would poison the cycle model, which is worse than a
  crash.
  """

  @doc """
  Parse a JSON string.

      iex> RiscGP.JSON.parse(~s({"a": 1, "b": [true, null]}))
      %{"a" => 1, "b" => [true, nil]}
  """
  def parse(source) when is_binary(source) do
    case parse_value(skip_space(source)) do
      {value, rest} ->
        case skip_space(rest) do
          "" -> value
          trailing -> raise ArgumentError, "trailing content after JSON value: #{inspect(binary_part(trailing, 0, min(40, byte_size(trailing))))}"
        end

      :error ->
        raise ArgumentError, "invalid JSON"
    end
  end

  @doc "Read and parse a file."
  def parse_file!(path), do: path |> File.read!() |> parse()

  # ---------------------------------------------------------------------------

  defp parse_value(<<"true", rest::binary>>), do: {true, rest}
  defp parse_value(<<"false", rest::binary>>), do: {false, rest}
  defp parse_value(<<"null", rest::binary>>), do: {nil, rest}

  defp parse_value(<<?", rest::binary>>), do: parse_string(rest, [])
  defp parse_value(<<?[, rest::binary>>), do: parse_array(skip_space(rest), [])
  defp parse_value(<<?{, rest::binary>>), do: parse_object(skip_space(rest), %{})

  # A number, including an optional leading minus and a fractional part. We do
  # not need exponents in the table, but accepting them costs two clauses and
  # avoids a surprising failure if one is ever added.
  defp parse_value(source) do
    case take_number(source, []) do
      {digits, rest} when digits != [] -> {to_number(digits), rest}
      _ -> :error
    end
  end

  defp parse_string(<<?", rest::binary>>, acc) do
    {acc |> Enum.reverse() |> List.to_string(), rest}
  end

  defp parse_string(<<?\\, ?", rest::binary>>, acc), do: parse_string(rest, ["\"" | acc])
  defp parse_string(<<?\\, ?\\, rest::binary>>, acc), do: parse_string(rest, ["\\" | acc])
  defp parse_string(<<?\\, ?/, rest::binary>>, acc), do: parse_string(rest, ["/" | acc])
  defp parse_string(<<?\\, ?b, rest::binary>>, acc), do: parse_string(rest, ["\b" | acc])
  defp parse_string(<<?\\, ?f, rest::binary>>, acc), do: parse_string(rest, ["\f" | acc])
  defp parse_string(<<?\\, ?n, rest::binary>>, acc), do: parse_string(rest, ["\n" | acc])
  defp parse_string(<<?\\, ?r, rest::binary>>, acc), do: parse_string(rest, ["\r" | acc])
  defp parse_string(<<?\\, ?t, rest::binary>>, acc), do: parse_string(rest, ["\t" | acc])
  defp parse_string(<<?\\, ?u, a, b, c, d, rest::binary>>, acc) do
    code = String.to_integer(<<a, b, c, d>>, 16)
    parse_string(rest, [<<code::utf8>> | acc])
  end

  defp parse_string(<<c::utf8, rest::binary>>, acc), do: parse_string(rest, [c | acc])
  defp parse_string(<<>>, _acc), do: :error

  defp parse_array(<<?], rest::binary>>, acc), do: {Enum.reverse(acc), rest}
  defp parse_array(<<?,, _rest::binary>>, _acc), do: :error

  defp parse_array(source, acc) do
    with {value, rest} <- parse_value(source),
         cont <- skip_space(rest) do
      case cont do
        <<?,, tail::binary>> -> parse_array(skip_space(tail), [value | acc])
        <<?], tail::binary>> -> {Enum.reverse([value | acc]), tail}
        _ -> :error
      end
    else
      _ -> :error
    end
  end

  defp parse_object(<<?}, rest::binary>>, acc), do: {acc, rest}
  defp parse_object(<<?,, _rest::binary>>, _acc), do: :error

  defp parse_object(<<?", rest::binary>>, acc) do
    with {key, after_key} <- parse_string(rest, []),
         <<?:, after_colon::binary>> <- skip_space(after_key),
         {value, after_value} <- parse_value(skip_space(after_colon)),
         cont <- skip_space(after_value) do
      acc = Map.put(acc, key, value)

      case cont do
        <<?,, tail::binary>> -> parse_object(skip_space(tail), acc)
        <<?}, tail::binary>> -> {acc, tail}
        _ -> :error
      end
    else
      _ -> :error
    end
  end

  defp parse_object(<<_c, _rest::binary>>, _acc), do: :error

  defp take_number(<<c, rest::binary>>, acc)
       when c in ~c"0123456789+-.eE" do
    take_number(rest, [c | acc])
  end

  defp take_number(source, acc), do: {acc |> Enum.reverse() |> List.to_string(), source}

  defp to_number(digits) do
    if String.contains?(digits, [".", "e", "E"]) do
      String.to_float(normalise_float(digits))
    else
      String.to_integer(digits)
    end
  end

  # `String.to_float/1` insists on a digit on both sides of the point, so `1.`
  # becomes `1.0` and `.5` becomes `0.5`. An exponent is passed through
  # untouched, since `1.0e5` is already well formed.
  defp normalise_float(digits) do
    case String.split(digits, ["e", "E"], parts: 2) do
      [mantissa] -> pad_mantissa(mantissa)
      [mantissa, exponent] -> pad_mantissa(mantissa) <> "e" <> exponent
    end
  end

  defp pad_mantissa("-" <> rest), do: "-" <> pad_mantissa(rest)

  defp pad_mantissa(mantissa) do
    cond do
      not String.contains?(mantissa, ".") -> mantissa <> ".0"
      String.ends_with?(mantissa, ".") -> mantissa <> "0"
      String.starts_with?(mantissa, ".") -> "0" <> mantissa
      true -> mantissa
    end
  end

  defp skip_space(<<c, rest::binary>>) when c in [?\s, ?\t, ?\n, ?\r], do: skip_space(rest)
  defp skip_space(source), do: source
end
