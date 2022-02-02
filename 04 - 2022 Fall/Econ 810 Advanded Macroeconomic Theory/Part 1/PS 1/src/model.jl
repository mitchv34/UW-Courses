using Parameters

include(tauchen.jl)

# Define the primitives of the model
@with_kw struct Primitives
    β       ::Float64 = 0.975 #discount rate
    r       ::Float64 = 0.04 #interest rate
    A_min   ::Float64 = 0 #Asset holdingss lower bound
    A_max   ::Float64 = 75.0 #Asset holdings upper bound
    nA      ::Int64 = 1000 #number of Asset holdings grid points
    A_grid  ::Array{Float64, 1} = range(A_min, length = nA, stop = A_max)) # Asset holdings grid
    γ       ::Float64 = 2 # CRRA Parameter
    U       ::Function = (C) -> (C**(1-γ))/(1 - γ) # Utility function
    nZ      ::Int64 = 5  # number of possible values for the persistent shock
    nε      ::Int64 = 5  # number of possible values for the transitory shock 
end

# Define the result structure of the model
mutable struct Results
    pol_fun ::Array{Float64, 3} # Policy function
    value_fun ::Array{Float64, 3} # Value function
    Π ::Array{Float64, 2} # Transition matrix for Persistent shock
    ε ::Array{Float64} # Transition matrix for Persistent shock
    μ       ::Array{Float64, 2}         # Distribution of assets
end

# Initialize the model
function initialize(ρ, σ_z, σ_ε)
    prim  = Primitives()
    pol_fun = zeros(prim.nA, prim.nZ, prim.nε)
    value_fun = zeros(prim.nA, prim.nZ, prim.nε)
    