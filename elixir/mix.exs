defmodule GPark.MixProject do
  use Mix.Project

  def project do
    [
      app: :gpark,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description: description(),
      package: package()
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
      links: %{
        "GitHub" => "https://github.com/scairesearch/gpark",
        "Changelog" => "https://github.com/scairesearch/gpark/blob/main/CHANGELOG.md"
      },
      # Only paths that exist *inside* this directory. `mix hex.publish` builds the
      # tarball from the mix project root, which is `elixir/`, so the repository's
      # README, LICENSE and CHANGELOG one level up cannot be listed here -- a glob
      # naming them silently matches nothing and ships a tarball with no licence in
      # it. The `licenses` metadata above is what the registry displays and enforces;
      # the canonical text lives at the repository root.
      files: ~w(lib .formatter.exs mix.exs)
    ]
  end
end
