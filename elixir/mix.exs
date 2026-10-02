defmodule GPark.MixProject do
  use Mix.Project

  def project do
    [
      app: :gpark,
      version: "0.1.0-dev",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description: description(),
      package: package(),
      docs: docs()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:jason, "~> 1.4"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end

  defp description do
    "Hand-written PTX kernels for quant workloads where nvcc and cuBLAS lose."
  end

  defp package do
    [
      licenses: ["AGPL-3.0"],
      links: %{"GitHub" => "https://github.com/gpark/gpark"},
      files: ~w(lib .formatter.exs mix.exs README* LICENSE* CHANGELOG*)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "docs/ARCHITECTURE.md",
        "docs/QUANT.md",
        "docs/PTX-SUBSET.md",
        "docs/VALIDATION.md",
        "docs/ROADMAP.md",
        "docs/DECISIONS.md",
        "docs/CONTEXT.md",
        "docs/GLOSSARY.md"
      ]
    ]
  end
end
