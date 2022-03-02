using Parameters, StatsBase

@with_kw struct Primitives
    # Parameters
    r       ::Float64           = 0.04                                                         # Risk-free interest rate
    β_lf    ::Float64           = 0.99                                                         # Discount factor for firms and lenders
    β       ::Float64           = 0.99                                                         # Discount factor
    δ       ::Float64           = 0.1                                                          # Job destruction rate
    ζ       ::Float64           = 1.6                                                          # Matching elasticity
    κ       ::Float64           = 0.995                                                        # Vacancy posting cost
    z       ::Float64           = 0.4                                                          # Something
    σ       ::Float64           = 2                                                            # CRRA parameter
    ## Probabilities
    p_L     ::Float64           = 0.5                                                          # Probability of lower human capital 
    p_H     ::Float64           = 0.05                                                         # Probability of higher human capital
    ## Grid Parameters
    T       ::Int64             = 100                                                          # Number of periods
    h_min   ::Float64           = 0.5                                                          # Minimum human capital
    h_max   ::Float64           = 1.5                                                          # Maximum human capital
    h_space ::Float64           = 0.01                                                         # Human capital grid spacing
    w_min   ::Float64           = 0.0                                                          # Minimum wage
    w_max   ::Float64           = 1.0                                                          # Maximum wage
    w_space ::Float64           = 0.01                                                         # Wage grid spacing
    
    # Grids
    h_grid  ::Array{Float64}    = h_min:h_space:h_max                                          # Human capital grid
    w_grid  ::Array{Float64}    = w_min:w_space:w_max                                          # Wage grid
    
    # Functions
    f       ::Function          = (h) -> h                                                     # Production function
    M       ::Function          = (u, v) -> u*v / (u^ζ + v^ζ)^(1/ζ)                            # Matching function
    H_u     ::Function          = (i) -> sample( [max(i-1, 1), i], Weights( [p_L , 1-p_L ] ) ) # Human capital process (unemployed)
    H_w     ::Function          = (i) -> sample( [min(nH, i+1), i], Weights( [p_H , 1-p_H ] )) # Human capital process (employed)
    H       ::Function          = (i, e) -> ( e == 1 ) ? H_e(i) : H_u(i)                       # Human capital process
    u       ::Function          = (c) -> ( c^(1-σ) )/( 1 - σ)                                  # Utility function
end