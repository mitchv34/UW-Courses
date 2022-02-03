using Parameters

include("tauchen.jl")

# Define the primitives of the model
@with_kw struct Primitives
    β       ::Float64 = 0.975 #discount rate
    r       ::Float64 = 0.04 #interest rate
    A_min   ::Float64 = 0 #Asset holdingss lower bound
    A_max   ::Float64 = 75.0 #Asset holdings upper bound
    nA      ::Int64 = 1000 #number of Asset holdings grid points
    A_grid  ::Array{Float64, 1} = range(A_min, length = nA, stop = A_max) # Asset holdings grid
    γ       ::Float64 = 2 # CRRA Parameter
    U       ::Function = (C) -> (C^(1-γ))/(1 - γ) # Utility function
    nP      ::Int64 = 11  # number of possible values for the persistent shock
    nε      ::Int64 = 5  # number of possible values for the transitory shock 

    T       ::Int64 = 35 # number of periods in the model
    # Todo: substitute with the Estimated values
    κ       ::Array{Float64, 1} = zeros(T) #persistent income

end

# Struct to hold the results of the model
mutable struct Results
    pol_fun ::Array{Float64, 3} # Policy function
    value_fun ::Array{Float64, 3} # Value function
    
    P :tationary_distributions( MarkovChain(ε_mat))[1] :Array{Float64} # Transitory shock values
    ε_prob ::Array{Float64, 1} # Probability of the transitory shock

    Y       ::Array{Float64, 3}     #  Income 
    μ       ::Array{Float64, 3}     # Distribution of assets

end

function initialize(ρ, σ_ζ, σ_ε)
    # Load primitives of the model
    prim  = Primitives()

    # Pre-allocate space for results
    pol_fun = zeros(prim.nA, prim.nP, prim.T)
    value_fun = zeros(prim.nA, prim.nP, prim.T)
    μ = zeros(prim.nA, prim.nP, prim.T)

    # Generate Permanent income and  shock transition matrix and values
    P, P_mat = tauchenMethod(0.0, σ_ζ, ρ, prim.nP)
    ε, ε_mat = tauchenMethod(0.0, σ_ε, 0.0, prim.nε)
    ε_prob = ε_mat[1, :]# Probability otationary_distributions( MarkovChain(ε_mat))[1] 
        for j in 1:prim.nε
            for i in 1:prim.nP
                    Y[i, j, t] = exp(prim.κ[t] + P[i] + ε[j])
            end
        end
    end

    # And we can pre-compuite last period value and policy functions
    pol_fun[:, :, :, prim.T] .= 0 # Last period policy function is 0 since agents consume everything
    # Calculate C
    for i in 1:prim.nA
        for j in 1:prim.nε
            for k in 1:prim.nP
                C = (1+prim.r)prim.A_grid[i] + Y[k, j,prim.T]
            end
        end 
    end
    
    value_fun[:, :,  :,  prim.T] = prim.U.( C ) # Last period value function is the utility of the final state
    
    # Define initial distribution
    μ = ones(prim.nA, prim.nP, prim.T) / (prim.nA * prim.nP)

    # Initialize resutlts
    # res = Results(pol_fun, value_fun, P, P_mat, ε, Y, μ)

    return prim, Results(pol_fun, value_fun, P, P_mat, ε, ε_prob, Y, μ)

end

# # Bellman operator

function Bellman_op(prim, res)
    @unpack nA, nP, nε, κ, T, U = prim
    @unpack pol_fun, value_fun, P, ε, Y, P_mat, ε_prob = res

    # Loop backwards over time
    for t ∈ T-1:-1:1
        # Loop over all possible values of persistent income
        for p_i ∈ 1:nP
            # Loop over all possible values of transitory shock
            p = P[p_i]
                # Calculate income today
                Y_today = Y[p_i, e_i, t]
                # Loop over all possible values of asset holdings
                for ai ∈ 1:nA
                    # Calculate consuption todday for every possible asset holdings tomorrow (given states)
                    C_today = (1+r)A_grid[ai] + Y_today .- A_grid
                    c_neg = C_today < 0
                    # Calculate utility of tomorrow given the current state (negative consumption excluded)
                    U_today = vcat( U.(C_today[c_neg]), repeat(Inf, sum(c_neg)) )
                    # Calculate value of tomorrow given the current state (negative consumption excluded)
                    Π_next = P_mat[p_i, :]
                    
                end
            end
        end # 
    end # T
end