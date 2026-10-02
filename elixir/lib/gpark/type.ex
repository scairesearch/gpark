defmodule Gpark.Type do
  @moduledoc """
  Two-level type system: **container** width and **element** format are separate.

  Taichi, the closest existing reference for a direct GPU language, does not have
  `bf16`, does not have fp8, and has no sub-byte primitives — which disqualifies
  it from essentially all tensor-core work. It separately has an arbitrary-width
  `quant` extension capped at 64 bits. So it half-solved sub-byte numerics and
  stopped.

  gpark separates the two concerns instead, because conflating them is what
  forces the choice between "has fp8" and "has 4-bit integers":

      container width   how many bits the machine moves at once (b1 b2 b4 b8
                        b16 b32 b64) — this is what PTX actually loads
      element format    what those bits mean (s4 u4 e2m1 e4m3 bf16 f32 …) — this
                        is what the algorithm computes on

  A packed value is therefore a `%Gpark.Type.Packed{}`: a container plus a
  logical element type plus a lane count. `:s4` has no PTX spelling because PTX
  has no 4-bit arithmetic, and pretending otherwise would produce kernels that
  `ptxas` rejects. Instead, sub-byte elements live *inside* a container and are
  extracted with shift/mask, or converted up to a native width first.

  ## Examples

      iex> Gpark.Type.width(:e4m3)
      8

      iex> Gpark.Type.native?(:e4m3)
      true

      # four signed 4-bit integers packed into one 16-bit container
      iex> p = Gpark.Type.packed(:b16, :s4, 4)
      iex> {Gpark.Type.container(p), Gpark.Type.lanes(p)}
      {:b16, 4}

      # a 4-bit integer has no native PTX type; it lowers to a b16 container
      iex> Gpark.Type.native?(:s4)
      false
      iex> Gpark.Type.ptx_type(Gpark.Type.packed(:b16, :s4, 4))
      :b16
  """

  alias Gpark.Type.Packed

  # ---------------------------------------------------------------------------
  # Native types: name => {storage_bits, kind, signedness}
  # ---------------------------------------------------------------------------
  #
  # `e4m3`/`e5m2` are fp8 (E4M3/E5M2), sm_89+.
  # `e2m3`/`e3m2`/`e2m1` are fp6/fp4 (Blackwell, sm_100+/sm_120+).
  # `e8m0` is the block-scaling exponent format, unsigned, no sign.
  #
  # Note the register classes are *not* the storage widths: PTX keeps fp8 and
  # fp4 values in `.b16` registers and converts from/to those, so `e4m3` lives
  # in the 32-bit `%r` bank via a b16 view. See `reg_class/1`.

  @native %{
    # integers
    s8: {8, :int, :signed},
    u8: {8, :int, :unsigned},
    s16: {16, :int, :signed},
    u16: {16, :int, :unsigned},
    s32: {32, :int, :signed},
    u32: {32, :int, :unsigned},
    s64: {64, :int, :signed},
    u64: {64, :int, :unsigned},
    # raw bit containers
    b1: {1, :bit, :unsigned},
    b2: {2, :bit, :unsigned},
    b4: {4, :bit, :unsigned},
    b8: {8, :bit, :unsigned},
    b16: {16, :bit, :unsigned},
    b32: {32, :bit, :unsigned},
    b64: {64, :bit, :unsigned},
    # floats
    f16: {16, :float, :signed},
    bf16: {16, :float, :signed},
    f32: {32, :float, :signed},
    f64: {64, :float, :signed},
    # fp8
    e4m3: {8, :float, :signed},
    e5m2: {8, :float, :signed},
    # fp6 / fp4
    e2m3: {6, :float, :signed},
    e3m2: {6, :float, :signed},
    e2m1: {4, :float, :signed},
    e8m0: {8, :float, :unsigned},
    # predicate
    pred: {32, :pred, :unsigned}
  }

  # s2/u2/s4/u4 are not PTX types at all, so they are deliberately absent from
  # @native -- that absence is exactly what stops the backend believing `u4` is
  # something it can do arithmetic on. They still carry {bits, kind, sign}, because
  # a packed element type that cannot answer width, kind or sign is not much of a
  # type: all three returned nil, which made `s4` indistinguishable from garbage.
  @sub_byte %{
    s2: {2, :int, :signed},
    u2: {2, :int, :unsigned},
    s4: {4, :int, :signed},
    u4: {4, :int, :unsigned}
  }

  # The native type a sub-byte element widens to before arithmetic.
  @widen %{
    s2: :s16,
    u2: :u16,
    s4: :s16,
    u4: :u16,
    e2m1: :f32,
    e4m3: :f32,
    e5m2: :f32,
    e2m3: :f32,
    e3m2: :f32,
    e8m0: :f32,
    bf16: :f32,
    f16: :f32,
    b1: :u32,
    b2: :u32,
    b4: :u32
  }

  @doc "All natively-supported PTX types, sorted."
  def native_names, do: @native |> Map.keys() |> Enum.sort()

  @doc """
  The `{bits, kind, sign}` description of a type, or `nil` if it has none.

  Checks `@native` first, then `@sub_byte`. Funnelling all three of `width/1`,
  `kind/1` and `sign/1` through one lookup means they cannot disagree about what a
  type is -- previously each consulted `@native` on its own, so every sub-byte type
  silently answered `nil` from all three.
  """
  def describe(type) when is_atom(type) do
    Map.get(@native, type) || Map.get(@sub_byte, type)
  end

  @doc "True when `type` has a direct PTX spelling."
  def native?(type) when is_atom(type), do: Map.has_key?(@native, type)
  def native?(%Packed{}), do: false

  @doc "Storage width of a native type in bits. Nil for packed types."
  def width(type) when is_atom(type) do
    case describe(type) do
      {bits, _kind, _sign} -> bits
      nil -> nil
    end
  end

  def width(%Packed{container: container}), do: width(container)

  @doc "One of `:int`, `:float`, `:bit`, `:pred`, or `:quant`."
  def kind(type) when is_atom(type) do
    case describe(type) do
      {_bits, kind, _sign} -> kind
      nil -> nil
    end
  end

  def kind(%Packed{}), do: :quant

  def float?(type) when is_atom(type), do: kind(type) == :float
  def float?(%Packed{}), do: false

  def int?(type) when is_atom(type), do: kind(type) == :int
  def int?(%Packed{}), do: false

  @doc ":signed or `:unsigned`."
  def sign(type) when is_atom(type) do
    case describe(type) do
      {_bits, _kind, sign} -> sign
      nil -> nil
    end
  end

  def sign(%Packed{elem: elem}), do: sign(elem)

  @doc """
  True for signed *integers*, including packed sub-byte ones.

  Deliberately not `sign/1 == :signed`: the native table records a sign for floats
  too, because `f32` has a sign bit. That is a different question from whether a
  type is a signed integer, and conflating them makes `signed_type?(:f32)` answer
  true -- which would then pick a signed comparison modifier for a float compare.
  """
  def signed_int?(%Packed{elem: elem}), do: signed_int?(elem)

  def signed_int?(type) when is_atom(type) do
    match?({_bits, :int, :signed}, describe(type))
  end

  @doc "True for unsigned *integers*, including packed sub-byte ones."
  def unsigned_int?(%Packed{elem: elem}), do: unsigned_int?(elem)

  def unsigned_int?(type) when is_atom(type) do
    match?({_bits, :int, :unsigned}, describe(type))
  end

  # ---------------------------------------------------------------------------
  # Packed (sub-byte / multi-lane) types
  # ---------------------------------------------------------------------------

  @doc "Build a packed type."
  def packed(container, elem, count)
      when is_atom(container) and is_atom(elem) and is_integer(count) do
    %Packed{container: container, elem: elem, count: count}
  end

  @doc "The native container a packed or native type is stored in."
  def container(%Packed{container: container}), do: container
  def container(type), do: type

  @doc "How many logical elements fit in the container."
  def lanes(%Packed{count: count}), do: count
  def lanes(type) when is_atom(type), do: 1

  @doc """
  Bit width of the logical element.

  For a packed element this is either a native width, a sub-byte width implied
  by the name (`:s4`, `:u2`, …), or, failing both, inferred from how many lanes
  share the container.

      iex> Gpark.Type.elem_width(Gpark.Type.packed(:b8, :s4, 2))
      4
  """
  def elem_width(%Packed{elem: elem, container: container, count: count}) do
    cond do
      native?(elem) -> width(elem)
      sub_byte_width(elem) -> sub_byte_width(elem)
      true -> div(width(container), count)
    end
  end

  def elem_width(type) when is_atom(type), do: width(type)

  @doc """
  Bit width of a non-native sub-byte integer name (`:s2`, `:u2`, `:s4`, `:u4`).

      iex> Gpark.Type.sub_byte_width(:u4)
      4
  """
  def sub_byte_width(type) do
    case Map.get(@sub_byte, type) do
      {bits, _kind, _sign} -> bits
      nil -> nil
    end
  end

  @doc "The PTX type a gpark type is stored and manipulated as."
  def ptx_type(%Packed{container: container}), do: container
  def ptx_type(type) when is_atom(type), do: if(native?(type), do: type, else: nil)

  @doc """
  The native type a sub-byte element should be widened to for arithmetic.

  Widening to the smallest type that holds every value of the element type
  avoids a redundant shift later, and keeps `s4`/`u4` in a single `.b16` bank.

      iex> Gpark.Type.widen(:u4)
      :u16
      iex> Gpark.Type.widen(:e2m1)
      :f32
  """
  def widen(%Packed{elem: elem}), do: widen(elem)
  def widen(type) when is_atom(type), do: Map.get(@widen, type, :u32)

  @doc "Every type gpark knows about, for exhaustive tests and golden generation."
  def all do
    Enum.sort(native_names() ++ Map.keys(@sub_byte))
  end
end
