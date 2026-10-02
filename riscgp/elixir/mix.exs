defmodule RiscGP.MixProject do
  use Mix.Project

  def project do
    [
      app: :riscgp,
      version: "0.1.0-draft",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description()
    ]
  end

  def application do
    [extra_applications: []]
  end

  # Deliberately zero dependencies. riscgp is a standalone project: it must not
  # couple itself to gpark's dependency set, and a JSON reader for one file is
  # cheaper to own than to depend on. See `RiscGP.JSON`.
  defp deps, do: []

  defp description do
    "RVGPU — a RISC-V GPU instruction set, emitter, validator and cycle model."
  end
end
