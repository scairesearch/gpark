defmodule RiscGP.Table do
  @moduledoc """
  The RVGPU typed opcode table, loaded from `isa/rvgpu-table.json`.

  The JSON file is the single source of truth. The Elixir and Python
  implementations both read it at runtime; neither re-declares the data. That
  split is deliberate — the *data* must not drift, but the *logic* that
  consumes it should be implemented twice so the two can be cross-checked
  (validation layer L0).

  The table is loaded at compile time via `@external_resource` and baked into
  the module, so emission never touches the filesystem. The file is still read
  at compile time on every build, so editing it forces a recompile.

  ## Status

  `"1.0-draft"`, `UNFROZEN`. Per the riscgp plan the ISA must not be frozen
  until the P0 study reports and the P1 gate selects Path A or Path B. Both
  datapaths are present and tagged by `path`, so selection is a query, not a
  rewrite. `frozen?/0` returning `false` is a live assertion, not a placeholder
  — `RiscGP.Emit` stamps every emitted kernel with the unfrozen marker so
  output can never be mistaken for a ratified ISA.
  """

  @table_path Path.expand("../../../isa/rvgpu-table.json", __DIR__)
  @external_resource @table_path

  @raw RiscGP.JSON.parse_file!(@table_path)

  @doc "The path of the JSON table, as loaded at compile time."
  def table_path, do: @table_path

  @doc "ISA name and version, e.g. `{\"RVGPU\", \"1.0-draft\"}`."
  def isa, do: {"RVGPU", @raw["version"]}

  @doc "True while the ISA is still a draft. The plan forbids freezing before the P1 gate."
  def frozen?, do: @raw["status"] == "FROZEN"

  @doc "The declared status string, e.g. `\"UNFROZEN\"`."
  def status, do: @raw["status"]

  @doc "Address spaces the ISA defines."
  def address_spaces, do: @raw["address_spaces"]

  @doc "Register file descriptors, keyed by file name, each with `count` and ABI."
  def registers, do: Map.delete(@raw["registers"], "$comment")

  @doc "Number of physical registers in a file, or 0 for an unknown file."
  def reg_count(file) when is_atom(file), do: reg_count(Atom.to_string(file))

  def reg_count(file) when is_binary(file) do
    case @raw["registers"][file] do
      nil -> 0
      spec -> spec["count"]
    end
  end

  @doc """
  ABI name for register `id` in `file`.

  Files with an explicit ABI list return the name from that list; files with an
  `abi_prefix` return the prefix followed by the integer id. Returns nil for an
  unknown file.
  """
  def reg_abi(file, id) do
    spec = @raw["registers"][file]

    cond do
      is_nil(spec) -> nil
      is_list(spec["abi"]) -> Enum.at(spec["abi"], id)
      true -> spec["abi_prefix"] <> Integer.to_string(id)
    end
  end

  # ---------------------------------------------------------------------------
  # Opcode lookup
  # ---------------------------------------------------------------------------

  # Built with an anonymous function rather than a private `spec/1` helper: a
  # module attribute is evaluated before the module body, so it cannot call a
  # function defined later in the same module. This is the same Elixir
  # restriction `Gpark.Ops` documents, and the reason that table is built at
  # runtime.
  @opcodes (
            @raw["opcodes"]
            |> Enum.map(fn raw ->
              {raw["name"],
               %{
                 name: raw["name"],
                 domain: raw["domain"],
                 path: raw["path"],
                 parts: raw["parts"],
                 dtypes: raw["dtypes"],
                 ndest: raw["ndest"],
                 nops: raw["nops"],
                 latency: raw["latency"],
                 throughput: raw["throughput"],
                 unit: raw["unit"],
                 sync: raw["sync"],
                 fence: raw["fence"],
                 spaces: Map.get(raw, "spaces"),
                 modifiers: Map.get(raw, "modifiers"),
                 comment: Map.get(raw, "$comment")
               }}
            end)
            |> Map.new()
          )

  @doc "Every opcode spec, keyed by name."
  def opcodes, do: @opcodes

  @doc "Look up one opcode spec, or nil."
  def op_spec(name) when is_binary(name), do: Map.get(@opcodes, name)

  @doc "Every opcode name, sorted."
  def names, do: @opcodes |> Map.keys() |> Enum.sort()

  @doc "Opcodes in a given execution domain, sorted."
  def by_domain(domain) when is_atom(domain), do: by_domain(Atom.to_string(domain))

  def by_domain(domain) do
    @opcodes
    |> Enum.filter(fn {_n, spec} -> spec.domain == domain end)
    |> Enum.map(&elem(&1, 0))
    |> Enum.sort()
  end

  @doc "Opcodes available on a given datapath (`:a`, `:b`, or `:ab` for both)."
  def by_path(:ab), do: names()

  def by_path(path) when path in [:a, :b] do
    tag = String.upcase(Atom.to_string(path))
    @opcodes |> Enum.filter(fn {_n, s} -> s.path == tag or s.path == "AB" end) |> Enum.map(&elem(&1, 0)) |> Enum.sort()
  end

  @doc """
  Which execution unit runs this opcode, or nil for an unknown opcode.

  This is the provenance the whole design hangs on: on Path B a kernel is
  mostly `core` instructions that do no arithmetic, and a scheduler that cannot
  tell "computes" from "tells something else to compute" will reorder across
  the only barrier that matters.
  """
  def unit(name), do: spec_field(name, :unit)
  def domain(name), do: spec_field(name, :domain)
  def path(name), do: spec_field(name, :path)

  @doc "Cycles from issue to the result being visible to the *issuing* core."
  def latency(name), do: spec_field(name, :latency)

  @doc "Cycles between back-to-back issues to the same unit."
  def throughput(name), do: spec_field(name, :throughput)

  @doc "True if the opcode orders prior memory operations and may not move across a barrier."
  def fence?(name), do: spec_field(name, :fence)

  @doc "True if the opcode participates in tile-synchronous execution."
  def sync?(name), do: spec_field(name, :sync)

  @doc "Permitted address spaces for a memory-addressed opcode, or nil."
  def spaces(name), do: spec_field(name, :spaces)

  @doc "Permitted modifiers, or nil. `[nil]` means the opcode takes no modifier."
  def modifiers(name), do: spec_field(name, :modifiers)

  @doc "Accepted data types, or `[]` when the opcode has no type suffix."
  def dtypes(name), do: spec_field(name, :dtypes)

  @doc "Destination count. 0 for stores, branches and synchronisation."
  def ndest(name), do: spec_field(name, :ndest)

  @doc "Operand count."
  def nops(name), do: spec_field(name, :nops)

  defp spec_field(name, field) do
    case Map.get(@opcodes, name) do
      nil -> nil
      spec -> Map.get(spec, field)
    end
  end
end
