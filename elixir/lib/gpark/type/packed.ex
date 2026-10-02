defmodule Gpark.Type.Packed do
  @moduledoc """
  A logical element type packed `count` times into a single `container`.

      %Gpark.Type.Packed{container: :b8, elem: :s4, count: 2}

  `elem` is a *logical* type and need not be native: `:s2`, `:u2`, `:s4`,
  `:u4` and the narrow floats all live here. `container` is always native.
  """
  @enforce_keys [:container, :elem, :count]
  defstruct [:container, :elem, :count]

  @type t :: %__MODULE__{container: atom(), elem: atom(), count: pos_integer()}
end
