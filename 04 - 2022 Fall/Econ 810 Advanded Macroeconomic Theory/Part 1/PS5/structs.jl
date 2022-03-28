using Parameters, DelimitedFiles, Distributed, SharedArrays
include("tauchen.jl")

# Primitives of the model
@with_kw struct test8
    
    # Parameters
    
    β               ::Float64               = (1/1.04)^6                        # Discount factor of agents
    r               ::Float64               = (1.04)^6                          # Risk-free interest rate
    κⱼ              ::Array{Float64, 1}                                         # Age profile of earnings
    
    # Grids
    ## Asset holding grid
    b_min           ::Float64             =  0.0                                # Lower bound
    b_max           ::Float64             = 15.0                                # Upper bound
    nb              ::Int64               = 30                                  # Number of points
    b_grid          ::Array{Float64, 1}   = range(b_min, b_max, length = nb)    # Asset holding grid
    
    ## (Adult) Human capital process h' = ρₕ*h + η where η ∼ N(0, σ_η)) 
    nₕ              ::Int64                = 10                                  # Number of grid points for human capital
    σ_η             ::Float64              = 2.0                                # Standard deviation of human capital shock
    ρₕ              ::Float64              = (0.97)^6                            # Persistence of human capital shock
    h_proc                                = tauchenMethod(0.0, sqrt(σ_η), ρₕ, nₕ) # Use Tauchen's method to generate the grid and transition matrix
    h_grid          ::Array{Float64}      = h_proc[1]                            # transition states
    Πₕ              ::Array{Float64,2}    = h_proc[2]                            # probabuilities of each state
    
    ## Child human capital process hc' = (1 - ω)*hc + κ*ω*i 
    ω               ::Float64             = 0.5                                 # Share of previous human capital
    κ               ::Float64             = 1/10                                # Importance of investment in child human capital
    hc_next         ::Function            = (hc, i) -> (1 - ω)*hc + κ*ω*i       # Child human capital update process


    # Time
    ## Time structure in this model is weird since we have different Bellman equations for different periods
    T           ::Array{Array{Float64}} = [[4], [5,6,7,8], [9,10,11,12], [13]] # [[new adult], [parets], [post-child], [retired]]

end

# Resutls of the model

# Value function
mutable struct ValueFunction
    # Value function
    ## Newly Independent Adults (j= 4)
    new_adult           ::SharedArray{Float64, 2}           # V₄ Value function for period 4, states (b,h)

    ## Parenting Stage (j= 5,6,7,8)
    parent              ::SharedArray{Float64, 4}           # Value function for period 5,6,7,8, states (b,h,hc,t)

    ## Post Child Working Stage (j= 9,10,11,12)
    post_child_1        ::SharedArray{Float64, 3}           # V₉ Value function for period 9 states (b,h,hc)
    post_child_next     ::SharedArray{Float64, 3}           # Value function for period 10,11,12 states (b,h,h4)

    ## Value at retirement (j= 13)
    retired             ::SharedArray{Float64, 2}           # Vₓ Value function for period 13, states (b,h)
end


# Policy Function
mutable struct PolicyFunctionAssets
    
    # Value function
    ## Newly Independent Adults (j= 4)
    new_adult           ::SharedArray{Float64, 2}           # V₄ Value function for period 4, states (b,h)

    ## Parenting Stage (j= 5,6,7,8)
    parent              ::SharedArray{Float64, 4}           # Value function for period 5,6,7,8, states (b,h,hc,t)

    ## Post Child Working Stage (j= 9,10,11,12)
    post_child_1        ::SharedArray{Float64, 3}           # V₉ Value function for period 9 states (b,h,hc)
    post_child_next     ::SharedArray{Float64, 3}           # Value function for period 10,11,12 states (b,h,h4)
    
end


mutable struct Resutls

    # Value Function
    value_function          ::ValueFunction 

    # Policy Function
    ## Assets
    pol_fun_b               ::PolicyFunctionAssets              # Policy function for asset holdings
    ## Investment 
    pol_fun_i               ::SharedArray{Float64, 4}           # Policy function for investmentin child human capital just for period 5,6,7,8
    ## Transfer
    pol_fun_τ               ::SharedArray{Float64, 3}           # Policy function for transfer just for period 9 

end