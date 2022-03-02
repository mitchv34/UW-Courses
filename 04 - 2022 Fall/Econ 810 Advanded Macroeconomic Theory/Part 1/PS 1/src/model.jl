using Parameters, Distributed, SharedArrays, ProgressBars, Distributions, StatsBase

include("tauchen.jl")

# Define the primitives of the model
# Primitive structure
@with_kw struct Primitives

    # life-cycle
    κ::Array{Float64, 1}      = κ # age profile of earnings from R code (in log income terms)
    T::Int64                  = length(κ)                               # lifespan

    # preference parameters
    γ::Float64                = 2.0                              # CRRA
    β::Float64                = 0.975                            # discount rate
    r                         = 0.04                             # interest rate

    # asset grid
    A_min::Float64            = 0.0                               # Lower bound
    A_max::Float64            = 50000.0                         # Upper bound
    nA::Int64                 = 1000                               # Number of points
    A_grid::Array{Float64, 1} = range(A_min, A_max; length = nA) 

    # transitory income shocks
    nε::Int64                 = 5                                # Number states
    σ_ε::Float64              = σ_ε                             # std dev 
    ε_proc                    = tauchenMethod(0.0, sqrt(σ_ε), 0.0, nε)    # transition matrix and states 
    ε_states::Array{Float64}  = ε_proc[1]                         # transition states
    ε_prob::Array{Float64}    = ε_proc[2][1,:]                    # probabuilities of each state

    # persistent income shocks
    nP::Int64                 = 5                                 # Number of states
    ρ_ζ::Float64              = ρ                                 # Persistence 
    σ_ζ::Float64              = σ_ζ                               # std dev 
    P_proc                    = tauchenMethod(0.0, sqrt(σ_ζ), ρ, nP)    # transition matrix and states 
    P_states::Array{Float64}  = P_proc[1]                         # transition states
    P_mat::Array{Float64,2}   = P_proc[2]                         # transition matrix

    # Functions
    u::Function               = (C) -> C^(1-γ) / (1-γ)            # utility function
    # Y::Function               = (κ,P,ε,t) -> exp(κp[t] + P + ε)   # income function
end
print(κ)
# Struct to hold the results of the model
mutable struct Results
    pol_fun ::SharedArray{Float64, 4} # Policy function
    value_fun ::SharedArray{Float64, 4} # Value function
    Y::Array{Float64, 3} # Income
end

function initialize()
    # Load primitives of the model
    prim  = Primitives()

    # Pre-allocate space for results
    pol_fun = SharedArray{Float64}(prim.nA, prim.nP, prim.nε, prim.T)
    value_fun = SharedArray{Float64}(prim.nA, prim.nP, prim.nε, prim.T)

    # Pre-compute all posible income states
    # First create a matrix of all sums of income shocks
    M = zeros(prim.nP, prim.nε)
    for i in 1:prim.nP
        for j in 1:prim.nε
            M[i,j] = prim.P_states[i] + prim.ε_states[j]
        end
    end
    # Then repeat the matrix to create a 3D matrix with all possible shocks each time period
    M = repeat(M, 1,1,prim.T)
    for t in 1:prim.T
        # add the age term to the matrix
        M[:,:,t] .+= prim.κ[t]
    end

    # Then pre-compute the income
    Y = exp.(M)

    # Pre-compuite last period value 
    # We can pre-Calculate the value function for the last period
    for i in 1:prim.nA
        for j in 1:prim.nP
            for k in 1:prim.nε
                y = Y[j,k,end]
                C = (1 + prim.r) * prim.A_grid[i] + y
                value_fun[i,j,k,end] = prim.u(C)
            end
        end
    end

    # Initialize resutlts
    res = Results(pol_fun, value_fun, Y)

    return prim, res

end

## Bellman operator

function Bellman_op(prim, res)
    @unpack nA, nP, nε, T, P_mat, ε_prob, r, A_grid, u, β  = prim
    @unpack pol_fun, value_fun, Y = res

    # Loop backwards over time
    for t ∈ ProgressBar(T-1:-1:1)
        lowest_ind = 0
        j_last = 1
        k_last = 1
        # println(t)
        # Loop over all possible values of 
        @sync @distributed for ijk in 1:(nA*nP*nε) # Iterate over assets and shocks
            i,j,k = Tuple(CartesianIndices((nA,nP,nε))[ijk])
            if (j != j_last) || (k != k_last)
                lowest_ind = 0
                j_last = j
                k_last = k
            end
            cand_max = -Inf
            i_A_max = 0
            y = Y[j,k,t] # Income at time t for asset i, shock j, and period t
            # Calculate consumption for each possible asset choice
            C = (1+r) * A_grid[i] + y .- A_grid
            possitive_C = findall(C .> 0)
            asset_choice_next_ind = [a_i for a_i in possitive_C if a_i >= lowest_ind]
            # asset_choice_next_ind = 1:nA
            for i_A_next ∈ asset_choice_next_ind
                # if C[i_A_next] <= 0
                #     continue
                # end
                util = u(C[i_A_next])
                exp_val = sum(res.value_fun[i_A_next,:,:,t+1] .* ε_prob' .* P_mat[j,:])
                val = util + β * exp_val
                if val >= cand_max
                    cand_max = val
                    i_A_max = i_A_next
                else
                    break
                end
            end
            # Update policy function
            pol_fun[i,j,k,t] = A_grid[i_A_max]
            # Update value function
            value_fun[i,j,k,t] = cand_max
            # lowest_ind = i_A_max
        end #
    end # T
end

## Simulation

# structure to hold simulation results:
mutable struct Sim_results
    C_sim ::Array{Float64, 2} # Consumption
    A_sim ::Array{Float64, 2} # Assets
    Y_sim ::Array{Float64, 2} # Income
end

function sim_data(nAgents::Int64, A_0::Float64, prim::Primitives, res::Results)

    @unpack A_grid, T, P_states, nP, P_mat, ε_states, nε, ε_prob, r, κ = prim

    A_0_i = findfirst(A_grid .== A_0)
    σ_ζ0 = 0.2
    P_0 = rand(Normal(0, σ_ζ0), nAgents)

    # Pre-allocate space for results
    C_agents = zeros(nAgents, T) # Consumption
    A_agents = zeros(nAgents, T) # Assets
    Y_agents = zeros(nAgents, T) # Income

    # Initialize the assets last period
    A_last = zeros(nAgents, 1)
    A_last .= A_0

    # First I to convert P_0  to admisible values of P_state
    P_last =  [ argmin(abs.(P_0[i] .- P_states))[1] for i in 1:nAgents]

    for t ∈ 1:T
        # Draw permanent income shock
        P_now = [sample(1:nP, Weights(P_mat[P_last[i],:])) for i in 1:nAgents]
        P_now_value = P_states[P_now]

        # Draw transitory income shock
        ε_now = sample(1:nε, Weights(ε_prob), nAgents) 
        ε_now_value = ε_states[ε_now]
        # Calculate income
        Y_now = exp.(P_now_value + ε_now_value .+  κ[t] )
        
        # Calculate asset holdings
        A_agents_ind = [findfirst(A_grid .== A_agents[i,t]) for i in 1:nAgents]

        if t < T
            A_now = [res.pol_fun[A_agents_ind[i], P_now[i], ε_now[i], 1] for i in 1:nAgents]
        else 
            A_now = zeros(nAgents, 1)
            A_now .= A_0
        end 
        
        # Calculate consumption
        C_now = (1+r) .* A_last .+ Y_now .- A_now

        for i in 1:nAgents
            c = C_now[i]
            if c < 0
                println("c = ", c, "y =  ", Y_now[i], "a_now = ", A_now[i], "a_last = ", A_last[i])
            end
        end

        A_agents[:,t] = A_now
        C_agents[:,t] = C_now
        Y_agents[:,t] = Y_now

        A_last = A_now       
        P_last = P_now

    end

    return Sim_results(C_agents, A_agents, Y_agents)
end