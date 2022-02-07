using Parameters

include("tauchen.jl")

# Define the primitives of the model
# Primitive structure
@with_kw struct Primitives

    # life-cycle
    κ::Array{Float64, 1}      = age_profile # age profile of earnings from R code (in log income terms)
    N::Int64                  = length(κ)                               # lifespan

    # preference parameters
    γ::Float64                = 2.0                              # CRRA
    β::Float64                = 0.975                            # discount rate

    # asset grid
    A_min::Float64            = 0.0                               # Lower bound
    A_max::Float64            = 2000000.0                         # Upper bound
    nA::Int64                 = 500                               # Number of points
    grid_a::Array{Float64, 1} = range(A_min, A_max; length = nA) 

    # transitory income shocks
    nε::Int64                 = 5                                 # Number states
    σ_ε::Float64              = sqrt(σ_ε)                         # std dev 
    ε_proc                    = tauchenMethod(0.0, σ_ζ, ρ, nε)    # transition matrix and states 
    ε_states::Array{Float64,1}= ε_proc[1]                         # transition states
    ε_prob::Array{Float64,2}  = ε_proc[2][1,:]                    # probabuilities of each state

    # persistent income shocks
    nζ::Int64                 = 5                                 # Number of states
    ρ_ζ::Float64              = ρ                                 # Persistence 
    σ_ζ::Float64              = sqrt(σ_ζ)                         # std dev 
    ζ_proc                    = tauchenMethod(0.0, σ_ζ, ρ, nζ)    # transition matrix and states 
    ζ_states::Array{Float64,1}= ζ_proc[1]                         # transition states
    ζ_mat::Array{Float64,2}   = ζ_proc[2]                         # transition matrix

    r                         = 0.04                              # exogeneous interest rate
end

# Struct to hold the results of the model
mutable struct Results
    pol_fun ::Array{Float64, 4} # Policy function
    value_fun ::Array{Float64, 4} # Value function
end

function initialize()
    # Load primitives of the model
    prim  = Primitives()

    # Pre-allocate space for results
    pol_fun = zeros(prim.nA, prim.nP, prim.nε, nprim.T)
    value_fun = zeros(prim.nA, prim.nP, prim.nε, prim.T)

    
    
    # Pre-compuite last period value and pAnd we can policy functions
    pol_fun[:, :, :, prim.T] .= 0 # Last period policy function is 0 since agents consume everything
    # Calculate C
    for i in 1:prim.nA
        for j in 1:prim.nε
            for k in 1:prim.nP
                C = (1+prim.r)prim.A_grid[i] + Y[k, j,prim.T]
            end
        end 
    end
    
#     value_fun[:, :,  :,  prim.T] = prim.U.( C ) # Last period value function is the utility of the final state
    
#     # Define initial distribution
#     μ = ones(prim.nA, prim.nP, prim.T) / (prim.nA * prim.nP)

#     # Initialize resutlts
#     # res = Results(pol_fun, value_fun, P, P_mat, ε, Y, μ)

#     return prim, Results(pol_fun, value_fun, P, P_mat, ε, ε_prob, Y, μ)

# end

# # # Bellman operator

# function Bellman_op(prim, res)
#     @unpack nA, nP, nε, κ, T, U = prim
#     @unpack pol_fun, value_fun, P, ε, Y, P_mat, ε_prob = res

#     # Loop backwards over timeζ_proc
#     for t ∈ T-1:-1:1
#         # Loop over all possible values of persistent income
#         for p_i ∈ 1:nP
#             # Loop over all possible values of transitory shock
#             p = P[p_i]
#                 # Calculate income today
#                 Y_today = Y[p_i, e_i, t]
#                 # Loop over all possible values of asset holdings
#                 for ai ∈ 1:nA
#                     # Calculate consuption todday for every possible asset holdings tomorrow (given states)
#                     C_today = (1+r)A_grid[ai] + Y_today .- A_grid
#                     c_neg = C_today < 0
#                     # Calculate utility of tomorrow given the current state (negative consumption excluded)
#                     U_today = vcat( U.(C_today[c_neg]), repeat(Inf, sum(c_neg)) )
#                     # Calculate value of tomorrow given the current state (negative consumption excluded)
#                     Π_next = P_mat[p_i, :]
                    
#                 end
#             end
#         end # 
#     end # T
# end