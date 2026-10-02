defmodule Gpark.Ops do
  @moduledoc """
  The typed opcode table.

  This is data, not IR, so it lives in its own module built at runtime. That is
  not just tidiness: a module attribute cannot call a local function, so an
  `@ops %{...}` built from helper calls simply does not compile. Building the
  table in a plain function body sidesteps that and keeps the entries readable as
  a flat list.

  Every entry is:

    * `parts` — the dotted components the emitter renders, in PTX order:
      `:name`, `:space`, `:sync`, `:modifier`, `:vec`, `:dtype`
    * `dtype` — accepted data/result types (a list; `[]` means "not applicable")
    * `ndest` — destination count (`0` for stores, branches, reductions)
    * `nops` — operand count
    * `otypes` — `:same` (operand type must equal `dtype`), `:any`, or a list
    * `spaces` — permitted address spaces, when the opcode is memory-addressed
    * `modifiers` — permitted modifiers, when the opcode has one; `[nil]` means
      the opcode takes no modifier
    * `sync` — true when the opcode participates in warp-synchronous execution

  Both `Gpark.IR` and `Gpark.PTX` read this table, which is what keeps
  validation and emission from drifting apart. See `docs/PTX-SUBSET.md` for what
  v0.1 actually covers and what is deliberately deferred.
  """

  # --- type groups ----------------------------------------------------------

  @int32 [:s32, :u32]
  @int64 [:s64, :u64]
  @int_all @int32 ++ @int64
  @floats [:f32, :f64]
  @mem_types @int_all ++ @floats ++ [:b64]

  # --- the table ------------------------------------------------------------

  # {name, parts, dtype, ndest, nops, otypes, extras}
  @specs [
    # data movement / conversion
    {"mov", [:name, :dtype], @mem_types ++ [:pred], 1, 1, :same, [sync: true]},
    {"cvt", [:name, :dtype], @mem_types ++ @floats, 1, 1, :any, []},

    # integer arithmetic
    {"add", [:name, :dtype], @mem_types, 1, 2, :same, [sync: true]},
    {"sub", [:name, :dtype], @mem_types, 1, 2, :same, [sync: true]},
    {"mul", [:name, :dtype], @mem_types, 1, 2, :same, [sync: true]},
    {"mad", [:name, :dtype], @int_all, 1, 3, :same, [sync: true]},
    {"neg", [:name, :dtype], @mem_types, 1, 1, :same, []},
    {"abs", [:name, :dtype], @int_all, 1, 1, :same, []},
    {"min", [:name, :dtype], @int_all ++ @floats, 1, 2, :same, [sync: true]},
    {"max", [:name, :dtype], @int_all ++ @floats, 1, 2, :same, [sync: true]},
    {"rem", [:name, :dtype], @int_all, 1, 2, :same, [sync: true]},
    {"div", [:name, :dtype], @int_all, 1, 2, :same, [sync: true]},
    {"shl", [:name, :dtype], @int_all, 1, 2, :same, [sync: true]},
    {"shr", [:name, :dtype], @int_all, 1, 2, :same, [sync: true]},
    {"and", [:name, :dtype], @mem_types, 1, 2, :same, [sync: true]},
    {"or", [:name, :dtype], @mem_types, 1, 2, :same, [sync: true]},
    {"xor", [:name, :dtype], @mem_types, 1, 2, :same, [sync: true]},
    {"not", [:name, :dtype], @mem_types ++ [:pred], 1, 1, :same, []},
    {"popc", [:name, :dtype], @int_all, 1, 1, :same, []},
    {"clz", [:name, :dtype], @int_all, 1, 1, :same, []},
    {"brev", [:name, :dtype], @int_all, 1, 1, :same, []},

    # 32-bit widening helpers — the backbone of 64-bit address arithmetic.
    {"mul.wide", [:name, :dtype], [:u32, :s32], 1, 2, [:u32, :s32], [sync: true]},
    {"mad.lo", [:name, :dtype], @int_all, 1, 3, :same, [sync: true]},
    {"mad.hi", [:name, :dtype], @int_all, 1, 3, :same, [sync: true]},

    # floating point
    # PTX has no `add.f.f32` — the `.f32` suffix already implies float, so these
    # are the same opcodes as the integer ones. Only `fma` needs a modifier.
    {"fma", [:name, :modifier, :dtype], @floats, 1, 3, @floats,
     [sync: true, modifiers: ["rn", "approx"]]},
    {"rcp", [:name, :dtype], [:f32, :f64], 1, 1, @floats, [sync: true]},
    {"rsqrt", [:name, :dtype], [:f32], 1, 1, @floats, [sync: true]},
    {"sqrt", [:name, :dtype], [:f32, :f64], 1, 1, @floats, [sync: true]},

    # comparison and predication
    {"setp", [:name, :modifier, :dtype], @mem_types, 1, 2, :same,
     [modifiers: ~w(eq ne lt le gt ge)]},
    {"selp", [:name, :dtype], @mem_types, 1, 3, :any, [sync: true]},
    {"slct", [:name, :dtype], @int_all, 1, 3, :same, [sync: true]},

    # memory
    {"ld", [:name, :space, :modifier, :vec, :dtype], @mem_types, 1, 1, :any,
     [
       spaces: [:global, :shared, :local, :const, :param],
       modifiers: [nil, "nc", "volatile", "cv"]
     ]},
    {"st", [:name, :space, :modifier, :vec, :dtype], @mem_types, 0, 2, :any,
     [spaces: [:global, :shared, :local, :param], modifiers: [nil, "wb", "cg", "cs", "wt"]]},
    {"prefetch", [:name, :space], [], 0, 1, :any, [spaces: [:global, :local]]},

    # atomics
    {"atom", [:name, :space, :modifier, :dtype], @mem_types, 1, 2, :any,
     [spaces: [:global, :shared], modifiers: ~w(add sub min max and or xor exch cas)]},
    {"red", [:name, :space, :modifier, :dtype], @mem_types, 0, 2, :any,
     [spaces: [:global, :shared], modifiers: ~w(add sub min max and or xor)]},

    # warp-level
    {"shfl", [:name, :sync, :modifier, :dtype], [:b32, :u32, :f32, :f64], 1, 3, :any,
     [modifiers: ~w(down up bfly idx)]},
    {"vote", [:name, :sync, :modifier, :dtype], [:pred, :u32], 1, 2, :any,
     [modifiers: ~w(any all ballot)]},
    {"activemask", [:name, :dtype], [:b32], 1, 0, :any, []},
    {"bar", [:name, :sync, :modifier], [], 0, 1, :any, [modifiers: [nil, "arrive", "red"]]},

    # control
    {"bra", [:name], [], 0, 1, [:label], []},
    {"brx", [:name, :modifier, :dtype], [], 0, 2, [:label], [modifiers: [nil, "uni"]]},
    {"ret", [:name], [], 0, 0, [], []},
    {"call.uni", [:name], [], 0, 0, [:label], []},
    {"exit", [:name], [], 0, 0, [], []},
    {"s2r", [:name, :dtype], @int_all, 1, 1, :sreg, []},
    {"cvta", [:name, :space, :dtype], @mem_types, 1, 1, :any,
     [spaces: [:global, :shared, :local, :const, :param, nil]]},
    {"nop", [:name], [], 0, 0, [], []}
  ]

  @type op_spec :: %{
          parts: [atom()],
          dtype: [atom()],
          ndest: non_neg_integer(),
          nops: non_neg_integer(),
          otypes: atom() | [atom()],
          spaces: [atom()] | nil,
          modifiers: [String.t() | nil] | nil,
          sync: boolean()
        }

  @doc "Build the full opcode table as a map of `name => spec`."
  def table do
    Map.new(@specs, fn {name, parts, dtype, ndest, nops, otypes, extras} ->
      base = %{
        parts: parts,
        dtype: List.wrap(dtype),
        ndest: ndest,
        nops: nops,
        otypes: otypes,
        spaces: nil,
        modifiers: nil,
        sync: false
      }

      {name, Map.merge(base, Map.new(extras))}
    end)
  end

  @doc "Look up one opcode spec, or nil."
  def fetch(name), do: Map.get(table(), name)

  @doc "Every opcode name in the table, sorted."
  def names, do: table() |> Map.keys() |> Enum.sort()

  @doc "Opcodes marked `sync: true`, i.e. warp-synchronous ones."
  def sync_ops,
    do: table() |> Enum.filter(fn {_n, s} -> s.sync end) |> Enum.map(&elem(&1, 0)) |> Enum.sort()

  @doc """
  Memory-addressed opcodes, which is what the scheduler must treat specially
  when it moves instructions across a `bar.sync`.
  """
  def memory_ops, do: ~w(ld st prefetch atom red)
end
