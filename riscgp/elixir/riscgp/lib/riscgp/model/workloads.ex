defmodule RiscGP.Model.Kernel do
  @moduledoc """
  One kernel of a workload, decomposed into path-independent quantities: MAC
  count, bytes read, bytes written.

  The decomposition carries no datapath and no product configuration, which is
  what lets the study swap Path A for Path B and swap nodes without touching the
  workload (plan: "swapping the datapath must not change NoC, SRAM, tile
  boundary, or the gpark IR").
  """

  defstruct [:role, :macs, :mac_kind, :read_bytes, :write_bytes]

  @type t :: %__MODULE__{
          role: atom(),
          macs: float(),
          mac_kind: :fp32 | :bf16 | :int8,
          read_bytes: float(),
          write_bytes: float()
        }

  @spec new(atom(), float(), atom(), float(), float()) :: t()
  def new(role, macs, mac_kind, read_bytes, write_bytes) do
    %__MODULE__{role: role, macs: macs, mac_kind: mac_kind, read_bytes: read_bytes, write_bytes: write_bytes}
  end

  @spec total_macs([t()]) :: float()
  def total_macs(kernels), do: Enum.reduce(kernels, 0.0, &(&1.macs + &2))

  @spec total_read_bytes([t()]) :: float()
  def total_read_bytes(kernels), do: Enum.reduce(kernels, 0.0, &(&1.read_bytes + &2))

  @spec total_write_bytes([t()]) :: float()
  def total_write_bytes(kernels), do: Enum.reduce(kernels, 0.0, &(&1.write_bytes + &2))

  @spec dominant_mac_kind([t()]) :: :fp32 | :bf16 | :int8
  def dominant_mac_kind(kernels) do
    kernels
    |> Enum.group_by(& &1.mac_kind, & &1.macs)
    |> Enum.max_by(fn {_kind, macs} -> Enum.sum(macs) end)
    |> elem(0)
  end

  @spec count([t()]) :: pos_integer()
  def count(kernels), do: length(kernels)
end

defmodule RiscGP.Model.Workload do
  @moduledoc """
  The four fixed P0 workloads. Every candidate is measured on exactly these.

  Plan P0.1 requires one named 7B-class decode workload, one named fused
  memory-bound kernel set, and one named training-shaped matmul, fixed before
  any candidate is chosen. One workload (WL1b) is added by documented scope
  amendment; see P0-DECISION.md. Bytes and MAC counts are derived here from the
  named model configurations so they can be checked by hand.
  """

  alias RiscGP.Model.Kernel

  defstruct [:id, :name, :track, :kind, :precision, :context, :kernels, :units, :notes]

  @type t :: %__MODULE__{
          id: String.t(),
          name: String.t(),
          track: String.t(),
          kind: :decode | :memory_bound | :training,
          precision: String.t(),
          context: non_neg_integer() | nil,
          kernels: [Kernel.t()],
          units: String.t(),
          notes: String.t()
        }

  @context 2048

  @doc "The fixed workload set, in the plan's order, with the amendment marked."
  @spec all() :: [t()]
  def all, do: [llama_8b_decode(), qwen_06b_decode(), fused_memory_bound(), training_gemm()]

  @spec decode() :: [t()]
  def decode, do: Enum.filter(all(), &(&1.kind == :decode))

  @spec find(String.t()) :: t()
  def find(id) do
    case Enum.find(all(), &(&1.id == id)) do
      nil -> raise KeyError, "no such workload: #{id}"
      workload -> workload
    end
  end

  def llama_8b_decode do
    hidden = 4096
    intermediate = 14336
    layers = 32
    kv_dim = 1024
    head_dim = 128
    heads = 32
    vocab = 128_256

    attn = 2 * hidden * hidden + 2 * kv_dim * hidden
    mlp = 3 * hidden * intermediate
    layer_weight_bytes = (attn + mlp) * layers
    lm_head_bytes = vocab * hidden
    kv_bytes = 2 * layers * @context * kv_dim * 2
    activation_bytes = 2 * layers * (hidden + intermediate) * 2 + vocab * 2
    attention_macs = 2.0 * layers * heads * @context * head_dim

    %__MODULE__{
      id: "wl1_llama31_8b_decode",
      name: "Llama-3.1-8B-Instruct, batch=1 greedy decode, 2048-token context",
      track: "S1",
      kind: :decode,
      precision: "int8 weights, fp16 activations, fp32 accumulate",
      context: @context,
      units: "one generated token",
      notes:
        "Plan P0.1 7B-class workload, kept unchanged. At batch=1 every weight is read " <>
          "exactly once per token, so this is a pure weight-streaming problem with no " <>
          "arithmetic reuse and no way to hide the memory system.",
      kernels: [
        Kernel.new(:layer_weight_stream, 2.0 * (layer_weight_bytes + lm_head_bytes), :int8, layer_weight_bytes + lm_head_bytes, 0.0),
        Kernel.new(:kv_stream, attention_macs, :fp32, kv_bytes, 0.0),
        Kernel.new(:activation_writeback, 0.0, :int8, 0.0, activation_bytes)
      ]
    }
  end

  def qwen_06b_decode do
    hidden = 1024
    intermediate = 3072
    layers = 28
    kv_dim = 1024
    head_dim = 128
    heads = 8
    vocab = 151_936

    attn = 4 * hidden * hidden
    mlp = 3 * hidden * intermediate
    layer_weight_bytes = (attn + mlp) * layers
    lm_head_bytes = vocab * hidden
    kv_bytes = 2 * layers * @context * kv_dim * 2
    activation_bytes = 2 * layers * (hidden + intermediate) * 2 + vocab * 2
    attention_macs = 2.0 * layers * heads * @context * head_dim

    %__MODULE__{
      id: "wl1b_qwen3_06b_decode",
      name: "Qwen3-0.6B, batch=1 greedy decode, 2048-token context",
      track: "S1",
      kind: :decode,
      precision: "int8 weights, fp16 activations, fp32 accumulate",
      context: @context,
      units: "one generated token",
      notes:
        "Scope amendment to plan P0.1, recorded in P0-DECISION.md and not a silent " <>
          "substitution. WL1 needs 7.77 GB of traffic per token, which no accessible node " <>
          "can serve inside an interactive latency budget; WL1b is the largest decode " <>
          "workload whose per-token stream an accessible node can actually serve, and " <>
          "it keeps the plan's batch=1, int8, fixed-context shape.",
      kernels: [
        Kernel.new(:layer_weight_stream, 2.0 * (layer_weight_bytes + lm_head_bytes), :int8, layer_weight_bytes + lm_head_bytes, 0.0),
        Kernel.new(:kv_stream, attention_macs, :fp32, kv_bytes, 0.0),
        Kernel.new(:activation_writeback, 0.0, :int8, 0.0, activation_bytes)
      ]
    }
  end

  def fused_memory_bound do
    hidden = 4096
    vocab = 128_256
    context = @context
    heads = 32
    head_dim = 128
    kv_dim = 1024

    %__MODULE__{
      id: "wl2_fused_int8_memory_bound",
      name: "fused int8 GEMV+RoPE, fused RMSNorm+requant, split-K decode reduction, int8 logit top-k",
      track: "S1,S2",
      kind: :memory_bound,
      precision: "int8 operands, fp16/fp32 accumulators",
      context: @context,
      units: "one invocation of the four-kernel set, one layer",
      notes:
        "Pre-registered negative control. Predicted to lose to any GPU by more than an " <>
          "order of magnitude because it is pure on-chip bandwidth and the accessible " <>
          "nodes give us narrow SRAM ports. Its job in the study is to prove the model " <>
          "reports losses, so the wins it produces can be trusted.",
      kernels: [
        Kernel.new(:gemv_rope, 1.0 * hidden * hidden, :int8, 1.0 * hidden * hidden, 1.0 * hidden * 2),
        Kernel.new(:rmsnorm_requant, 1.0 * hidden * hidden, :fp32, 1.0 * hidden * hidden * 2, 1.0 * hidden * hidden),
        Kernel.new(:splitk_reduce, 2.0 * heads * context * head_dim, :fp32, 2.0 * context * kv_dim * 2, 1.0 * heads * context * 4),
        Kernel.new(:logit_topk, 0.0, :int8, 1.0 * vocab * 2, 1.0 * vocab * 4)
      ]
    }
  end

  def training_gemm do
    n = 4096

    %__MODULE__{
      id: "wl3_training_bf16_gemm_4096",
      name: "bf16 4096^3 GEMM set: forward, gradient wrt activations, gradient wrt weights",
      track: "S2",
      kind: :training,
      precision: "bf16 operands, fp32 accumulate",
      context: nil,
      units: "three 4096^3 GEMMs",
      notes:
        "Pre-registered negative control and scope boundary. It quantifies how far the " <>
          "datacenter track is from training-class throughput so that no training claim " <>
          "can be made by accident.",
      kernels: [
        Kernel.new(:fwd_gemm, 2.0 * n * n * n, :bf16, 2.0 * n * n * 2, 1.0 * n * n * 2),
        Kernel.new(:dgrad_gemm, 2.0 * n * n * n, :bf16, 2.0 * n * n * 2, 1.0 * n * n * 2),
        Kernel.new(:wgrad_gemm, 2.0 * n * n * n, :bf16, 2.0 * n * n * 2, 1.0 * n * n * 2)
      ]
    }
  end
end