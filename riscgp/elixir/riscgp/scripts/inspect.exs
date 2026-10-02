alias RiscGP.Model.{Kernel, Workload}

for w <- Workload.all() do
  IO.puts(
    "#{w.id}: read=#{Float.round(Kernel.total_read_bytes(w.kernels) / 1.0e6, 2)} MB " <>
      "write=#{Float.round(Kernel.total_write_bytes(w.kernels) / 1.0e6, 3)} MB " <>
      "macs=#{Float.round(Kernel.total_macs(w.kernels) / 1.0e9, 3)} G " <>
      "kind=#{Kernel.dominant_mac_kind(w.kernels)} kernels=#{Kernel.count(w.kernels)}"
  )
end

params = RiscGP.Model.Params.all()
for product <- [:edge, :dc], node <- [:sky130, :cgf22] do
  IO.puts(
    "#{product}.#{node}: tiles=#{RiscGP.Model.Fabric.tiles(params, product, node)} " <>
      "clock=#{RiscGP.Model.Fabric.clock_hz(params, product, node) / 1.0e6} MHz " <>
      "sram=#{RiscGP.Model.Fabric.sram_kb_total(params, product, node)} kB " <>
      "die_read=#{Float.round(RiscGP.Model.Fabric.die_sram_read_bytes_per_cycle(params, product, node))} B/cyc " <>
      "dram=#{Float.round(RiscGP.Model.Fabric.dram_bytes_per_cycle_effective(params, product, node), 2)} B/cyc"
  )
end
