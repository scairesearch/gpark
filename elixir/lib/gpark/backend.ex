defmodule Gpark.Backend do
  @moduledoc """
  The contract a gpark backend implements, and the gate that enforces it.

  This is where gpark takes the Taichi lesson about a neutral IR and stops
  *claiming* to have one. The IR was already neutral in the sense that nothing in it
  knows about PTX, but nothing stopped a backend from quietly emitting a slower
  sequence for something it did not support. A kernel would run, produce the right
  answer, and cost three times what it should — and the only symptom would be a
  benchmark nobody could explain.

  `require!/2` turns that into a build-time failure naming the exact opcode or type
  the backend cannot handle. A backend is expected to be unable to do things; gpark's
  job is to make it say so rather than to approximate.

  ## Usage

      {:ok, ptx} = Gpark.Backend.require!(Gpark.PTX, kernel)
      text = Gpark.PTX.emit(ptx)

  or, if the kernel is already known to be within the backend's capabilities,
  `emit!/2`, which raises instead of returning `{:error, _}`.
  """

  alias Gpark.IR

  @typedoc "A problem that stops a backend handling a kernel."
  @type issue ::
          {:unsupported_op, binary()}
          | {:unsupported_type, atom()}

  @typedoc "A backend implementation."
  @type t :: module()

  @doc "A short name for the backend, for diagnostics."
  @callback name() :: atom()

  @doc "Opcode base names this backend can emit."
  @callback ops() :: [binary()]

  @doc "Types this backend can represent."
  @callback types() :: [atom()]

  @doc "Emit `kernel` as text."
  @callback emit(map()) :: binary()

  @doc """
  Check the kernel's own invariants, independently of this backend's capabilities.

  Returning `{:error, issues}` for a structurally invalid kernel is distinct from
  `require!/2` refusing a valid kernel this backend cannot handle.
  """
  @callback check(map()) :: {:ok, map()} | {:error, [term()]}

  @doc "Every opcode base used by `kernel`, across all blocks and terminators."
  @spec required_ops(map()) :: MapSet.t(binary())
  def required_ops(kernel) do
    for instr <- instrs(kernel), into: MapSet.new(), do: instr.base
  end

  @doc "Every type `kernel` mentions, including in operands and parameter declarations."
  @spec required_types(map()) :: MapSet.t(atom())
  def required_types(kernel) do
    from_instrs =
      for instr <- instrs(kernel),
          type <- [instr.dtype, instr.srctype],
          # `nil` is an atom in Elixir, and an instruction with no type -- `ret`,
          # `bra` -- leaves these nil. Filtering on is_atom/1 alone reports every
          # terminator as an unsupported type named nil.
          is_atom(type) and not is_nil(type),
          into: MapSet.new() do
        type
      end

    from_operands =
      for instr <- instrs(kernel),
          {type, _class, _id} <- IR.instr_regs(instr),
          into: MapSet.new() do
        type
      end

    from_params =
      for %{type: type} <- kernel.params, into: MapSet.new(), do: type

    MapSet.union(from_instrs, MapSet.union(from_operands, from_params))
  end

  @doc "Whether `backend` can emit `base`."
  @spec supports_op?(t(), binary()) :: boolean()
  def supports_op?(backend, base), do: base in backend.ops()

  @doc "Whether `backend` can represent `type`."
  @spec supports_type?(t(), atom()) :: boolean()
  def supports_type?(backend, type), do: type in backend.types()

  @doc """
  Check that `backend` can handle `kernel`, returning the kernel or every reason it
  cannot.

  Returns `{:ok, kernel}` unchanged rather than a transformed kernel: this is a gate,
  not a lowering step, so a caller that gets `{:ok, _}` knows nothing was rewritten on
  the way through.
  """
  @spec require!(t(), map()) :: {:ok, map()} | {:error, [issue()]}
  def require!(backend, kernel) do
    supported_ops = backend.ops()
    supported_types = backend.types()

    missing_ops =
      for base <- Enum.sort(required_ops(kernel)),
          base not in supported_ops,
          do: {:unsupported_op, base}

    missing_types =
      for type <- Enum.sort(required_types(kernel)),
          type not in supported_types,
          do: {:unsupported_type, type}

    case missing_ops ++ missing_types do
      [] -> {:ok, kernel}
      issues -> {:error, issues}
    end
  end

  @doc """
  `require!/2` then emit, raising if the backend cannot handle the kernel.

  The raise is deliberate. A missing capability is a programming error that should
  stop the run rather than produce output that is slower than expected and still
  correct.
  """
  @spec emit!(t(), map()) :: binary()
  def emit!(backend, kernel) do
    case require!(backend, kernel) do
      {:ok, checked} -> backend.emit(checked)
      {:error, issues} -> raise_unsupported(backend, kernel, issues)
    end
  end

  defp raise_unsupported(backend, kernel, issues) do
    detail =
      issues
      |> Enum.map(fn
        {:unsupported_op, base} -> "opcode #{base}"
        {:unsupported_type, type} -> "type #{inspect(type)}"
      end)
      |> Enum.join(", ")

    raise ArgumentError,
          "#{inspect(backend)} cannot emit #{kernel.name}: it has no #{detail}. " <>
            "Emit a slower sequence explicitly, or extend the backend."
  end

  defp instrs(kernel) do
    Enum.flat_map(kernel.blocks, &(&1.instrs ++ List.wrap(&1.term)))
  end
end
