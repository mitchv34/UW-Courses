using Parameters, LinearAlgebra, Distributions, QuantEcon, FixedEffectModels, LaTeXStrings
using Plots, DataFrames
theme(:vibrant) 
default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

# Structure to hold the parameters of the model
@with_kw struct Primitives
    # Parameters of the model
    ## From the paper
    α_K                     ::Float64=0.3           # Share of capital
    α_L                     ::Float64=0.65          # Share of labor
    δ                       ::Float64=0.145         # Depreciation rate
    ρ                       ::Float64=0.762         # Persistence of productivity
    σ                       ::Float64=0.0352        # Standard deviation of productivity shocks
    β                       ::Float64= 1/1.065      # Discount factor
    H                       ::Float64=0.6           # Preference of leisure

    ## Not from the paper
    A                       ::Float64=1.0           # Aggregate productivity
    f_p                     ::Float64=0.01         # Fixed cost of production

    # Grids
    nZ                      ::Int64 = 10     # Number of productivity shocks
    # Capital grid
    k_min                   ::Float64 = 0.0  # Minimum value of capital
    k_max                   ::Float64 = 5.0 # Maximum value of capital
    nK                      ::Int64 = 51    # Number of points in the grid
    k_grid::Array{Float64} = range(k_min, stop = k_max, length = nK) # Grid of capital
    
    #  Optimal decision rules
    ℓ_optim  = ( k, z, w ) ->  max( ( w / ( α_L * A * exp(z) * k^α_K ) )^(1/(α_L - 1)), 0 )
    F = (k, z, w) -> A * exp(z) .* k^α_K * ℓ_optim.(k, z, w)^α_L
    Π  = (k, z, w) -> F.(k, z, w) - ℓ_optim(k, z, w) * w - f_p
    ι  = (k, k_next) -> k_next - (1 - δ) * k

    # Wage grid
    w_min                   ::Float64 = 0.01
    w_max                   ::Float64 = 1.5


    # # Limits for the mass of entrants
    B_min       ::Float64 = 0.0
    B_max       ::Float64 = 1000.0

end




# Structure to hold the results of the model
mutable struct Results
    # Results of the model
    # Stochastic productivity process
    z_vals# ::Array{Float64, 1} # Productivity shocks
    Δ #:: Array{Float64, 2} # Markov transition matrix

    p_val   ::Array{Float64, 2} # Firm value given state variables
    n_opt   ::Array{Float64, 2} # Optimal labor demand for each possible state
    x_opt   ::Array{Float64, 2} # Optimal firm exit decicion for each state 
    k_opt   ::Array{Float64, 2} # Optimal capital decicion for each state
    w       ::Float64 # Market clearing wage
    μ       ::Array{Float64, 2} # Distribution of Firms
    B       ::Float64 # Mass of entrants
    λ₀      ::Float64 # Floatation cost of external finance
    λ₁      ::Float64 # Variational cost of external finance
    λ       ::Function # Exogenous cost of external finance 
    # Aditional results
    CDF   ::Array{Float64, 2} # CDF of productivity shocks
    I_mat ::Array{Float64, 2} # Investment decisions
    Π_mat ::Array{Float64, 2} # Firm proffit
end

# Tauchen's method to generate Markov transition matrix from AR(1) process
function tauchenMethod(μ::Float64, σ:: Float64, ρ::Float64, z_num::Int64; q::Int64=3, tauchenoptions=0, dshift=0) 
    
    # Tauchen's method for generating Markov chains. 
    # This is based on TauchenMethod.m from the VFIToolkit-matlab toolbox by Robert Kirkby
    # https://github.com/vfitoolkit/VFIToolkit-matlab

    # Create z_vals vector and transition matrix for the discrete markov process approximation of AR(1) process z'=μ+ρ*z+e, e~N(0,σ²), by Tauchens method
    # Inputs
    #   μ              - AR(1) process z'=μ+ρ*z+ε, ε~N(0,σ²)
    #   ρ              - AR(1) process z'=μ+ρ*z+ε, ε~N(0,σ²)
    #   σ²              - AR(1) process z'=μ+ρ*z+ε, ε~N(0,σ²)
    #   q              - max number of std devs from mean (default=3)
    #   z_num           - number of z_vals in discretization of z (must be an odd number)
    # Optional Inputs
    # Outputs
    #   z_vals         - column vector containing the z_num states of the discrete approximation of z
    #   Π    - transition matrix of the discrete approximation of z;
    #                    Π(i,j) is the probability of transitioning from state i to state j
    #
    # Helpful info: Π
    #   Var(z)=σ²/(1-ρ^2); note that if μ=0, then σ²z=σ²/(1-ρ^2).
    ###############
    
    
    if z_num==1
        z_vals=[μ/(1-ρ)]; #expected value of z
        Π=[1];

        return z_vals, Π
    end

        # σ = sqrt(σ²); #stddev of ε
        z_star = μ/(1-ρ) #expected value of z
        σ_z = σ/sqrt(1-ρ^2) #stddev of z
        z = z_star*ones(z_num, 1) + range(-q*σ_z, stop = q*σ_z, length = z_num)  
        ω = z[2] - z[1] #Note that all the points are equidistant by construction.
        
        zi=z*ones(1,z_num);

        zj=dshift*ones(z_num,z_num)+ones(z_num,1)*z'
        
        dist = Normal(μ, σ)

        P_part1 = cdf( dist, zj .+ ω/2-ρ * zi)
        P_part2 = cdf( dist, zj .- ω/2-ρ * zi)
        
        P = P_part1 - P_part2
        P[:, 1] = P_part1[:, 1]
        P[:, z_num] = 1 .- P_part2[:,z_num]
        

        z_vals=z;
        Π=P; #(z,zprime)
        return z_vals, Π
end

# Function to initialize the model
function initialize(λ₀::Float64, λ₁::Float64)
    # Initialize Primitives
    prim = Primitives()

    # Use Tauchen's method to generate productivity shocks markov chain and transition matrix
    z_vals, Δ =tauchenMethod(0.0, prim.σ, prim.ρ, prim.nZ, q=4)
    
    λ  = (Π, ι) -> (Π ≥ ι) ? 0 : λ₀ + λ₁ * (ι - Π)

    # Initialize results variables
    p_val = zeros(prim.nZ, prim.nK)
    n_opt = zeros(prim.nZ, prim.nK)
    x_opt = zeros(prim.nZ, prim.nK)
    k_opt = zeros(prim.nZ, prim.nK)
    w = (prim.w_max + prim.w_min)/2
    μ = ones(prim.nZ, prim.nK)  # Uniform distribution is the initial guess
    B = 1.0

    CDF = zeros(prim.nZ, prim.nK)
    I_mat = zeros(prim.nZ, prim.nK)
    Π_mat = zeros(prim.nZ, prim.nK)
    # Initialize Results
    res = Results(z_vals, Δ, p_val, n_opt, x_opt, k_opt, w, μ, B,λ₀, λ₁, λ, CDF, I_mat, Π_mat)
    # Retunr Primitives and Results
    return prim, res
end

# Bellman operator for W
function p_Bellman(prim::Primitives, res::Results)
    @unpack Π, nZ, ι, k_grid, δ, β, nK = prim
    @unpack w, z_vals, Δ, p_val, λ = res 

    temp_p = zeros(nZ, nK)
    temp_x =  zeros(nZ, nK)
    temp_k =  zeros(nZ, nK)

    # Iterate over all possible states
    for z_i ∈ 1:nZ
        z = z_vals[z_i]
        profit_state = Π.(k_grid, z, w) # Profit for each possible state and capital levels
        # Iterate over all possible capital levels
        for k_i ∈ 1:nK
            k = k_grid[k_i] # Current capital level
            Πₜ = profit_state[k_i] # Profit for the current state and capital level
            ιₜ = ι.(k, k_grid) # Investment in the next period for any choice of k_next given k
            λₜ = λ.(Πₜ, ιₜ) # Variational cost of external finance for each choice of k_next given k
            exp_cont_value = [Δ[z_i, :]' * res.p_val[:, i] for i in 1:nK] # Expected continuation value for each choice of k_next given k
            exp_no_cont_value = (1 - δ) * k_grid  # "Expected" value of not continuing given k
            # Bellman operator
            value = Πₜ .- (ιₜ + λₜ)  + β * maximum(hcat(exp_no_cont_value, exp_cont_value), dims=2) 
            # Find the optimal choice of k_next and x given k
            ind_max = argmax(value)
            temp_p[z_i, k_i] = value[ind_max]
            temp_x[z_i, k_i] = exp_cont_value[ind_max] ≥ exp_no_cont_value[ind_max]
            temp_k[z_i, k_i] = k_grid[ind_max]
        end
    end
    # Update results
    res.p_val = temp_p
    res.x_opt = temp_x
    res.k_opt = temp_k
end # p_Bellman

#Value function iteration for p_Bellman operator
function Tp_Bellman_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4)
    
    n = 0 #counter
    err = 100.0 #initialize error
    while  (err > tol) & (n < 4000)#begin iteration
        p_val_old = copy(res.p_val)
        p_Bellman(prim, res)
        err = maximum(  abs.(p_val_old - res.p_val ) ) #reset error level
        n+=1
        if n % 10 == 0
            # println("Iter =", n , " Error = ", err)
        end
    end
    println("Iter =", n , " Error = ", err)
end # Tp_Bellman_iterate

# Iterate until the labor market clears to finnd w
function solve_wage(prim::Primitives, res::Results; tol::Float64 = 1e-3,  n_max::Int64 = 1000)
    w_min = prim.w_min
    w_max = prim.w_max

    θ = 0.75
    n = 0
    
    while n < n_max
        Tp_Bellman_iterate(prim, res)

        EC = sum(res.p_val[:, 1])
        
        if n % 1 == 0
            println(n+1, " iterations; EC = ", EC, ", w = ", res.w, ", w_min = ", w_min, ", w_max = ", w_max, ", θ = ", θ)
        end
        if abs( EC ) < tol
            println("Wage converged in $(n+1) iterations, w = $(res.w)")
            break
        end

        if abs(EC) > 10
            θ = 0.5
        elseif abs(EC) > 0.5
            θ = 0.75
        elseif abs(EC) > 0.1
            θ = 0.9
        elseif abs(EC) > 0.01
            θ = 0.99
        end

        # adjust wage toward bounds according to tuning parameter
        if EC < 0
            w_old = res.w
            res.w = θ*res.w + (1-θ)*w_min
            w_max = w_old
        else
            w_old = res.w
            res.w = θ*res.w + (1-θ)*w_max
            w_min = w_old
        end
        
        n += 1
        
    end
end

function T_star_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, max_iter::Float64=Inf)

    @unpack nK, nZ, k_grid = prim
    @unpack z_vals, k_opt, x_opt, Δ = res

    k_pf = k_opt'
    x_pf = x_opt'

    transition_matrix = zeros(nK * nZ, nK * nZ)

    for (i_k, k) in enumerate(k_grid), (i_z, z) in enumerate(z_vals)
        for (i_k_p, k_p) in enumerate(k_grid), (i_z_p, z_p) in enumerate(z_vals)

            if k_p == k_pf[i_k, i_z]
                row = i_k + nK * (i_z - 1)
                col = i_k_p + nK * (i_z_p - 1)

                transition_matrix[row, col] += x_pf[i_k, i_z] * Δ[i_z, i_z_p] 
            end
        end
    end

    transition_matrix_entrant = zeros(nZ, nK * nZ )

    for (i_z, z) in enumerate(z_vals)
        for (i_k_p, k_p) in enumerate(k_grid), (i_z_p, z_p) in enumerate(z_vals)
            if k_p == k_pf[1, i_z]
                row = i_z
                col = i_k_p + nK * (i_z_p - 1)

                transition_matrix_entrant[row, col] += x_pf[1, i_z] * Δ[i_z, i_z_p] 
            end
        end
    end    

    M = res.μ'
    μ = reshape(M, nK * nZ, 1)
    μ_p = zeros(nK * nZ, 1)

    err = 100.0
    while err > 0.001
        # again we assume B = 1
        μ_p = transition_matrix' * μ + transition_matrix_entrant' * ones(nZ, 1) / nZ 

        err = maximum(abs.(μ_p .- μ))
        μ = μ_p
    end

    res.μ = reshape(μ, nK, nZ)'

end

function solve_mass_entrants(prim::Primitives, res::Results)


    # Solve for the stationary distribution with mass of entrants equal to 1
    res.B = 1.0
    T_star_iterate(prim, res)

    # Unpack values
    @unpack H, ℓ_optim, k_grid, F, f_p, α_L, α_K, A, nZ, nK = prim
    @unpack w, z_vals, μ, x_opt  = res

    # Compute the optimal labor demand and profits
    L_D = zeros(nZ, nK)
    P = zeros(nZ, nK)
    for i ∈ 1:nZ
        for j ∈ 1:nK
            L_D[i,j] = ℓ_optim.( k_grid[j], z_vals[i], w)
            P[i,j] =  (A * exp(z_vals[i]) .* k_grid[j]^α_K * L_D[i,j]^α_L - L_D[i,j] * w - f_p) 
        end
    end

    agg_L_D = sum(L_D .* μ .* x_opt)
    agg_P = sum(P .* μ .* x_opt)

    # Exploit homogeneity to get mass of entrants from first order conditions
    res.B = w / ( H * ( w * agg_L_D - agg_P ) )
    println("Mass of entrants = ", res.B)
    # Update distribution
    res.μ = res.B * res.μ
end


# Get aggregate results and generate tables
function get_agg_results(prim::Primitives, res::Results; save_tables::Bool = false)
    @unpack nZ, nK, k_grid, F, f_p, α_L, α_K, A, ℓ_optim, δ = prim
    @unpack w, μ, x_opt, B, k_opt, z_vals, p_val, λ₀, λ₁, = res
    # Other results
    k_matrix = (k_grid * ones(nZ)')'
    z_matrix = (ones(nK) * z_vals' )'

    # matrix results
    CDF = cumsum(μ; dims = 2)./sum(μ; dims = 2)

    I_mat = k_opt .- (1 - δ) .* k_matrix
    L_mat = ℓ_optim.(k_matrix, z_matrix, w)
    Y_mat = A  *(k_matrix.^α_K .*  L_mat.^α_L) .* exp.(z_matrix)
    Π_mat = Y_mat .- w * L_mat .- f_p

    Λ_mat = (I_mat .- Π_mat .> 0) .* (λ₀.+ λ₁ .* (I_mat .- Π_mat))

    d_pf = Π_mat .- I_mat .- Λ_mat

    # aggregate moments
    Y = sum(  Y_mat .* x_opt .* μ) -  B * f_p / nZ
    I = sum(  I_mat .* x_opt .* μ) + B * sum( k_opt[1, :] ) / nZ
    Λ = sum( Λ_mat .* x_opt  .* μ) + B * sum( Λ_mat[1, :])
    production_cost = sum(μ .* f_p)
    floatation_cost = sum( (Λ_mat .> 0) .* λ₀ .* μ)

    # cross-sectional moments

    M = sum(μ)

    size_mean = 1/abs(M - B) * sum(k_matrix .* μ .* x_opt)
    i_k_mean = 1/abs(M - B) * sum((k_matrix .> 0) .* (I_mat ./ k_matrix) .* μ .* x_opt)
    i_k_std = sqrt(1/abs(M - B) * sum((k_matrix .> 0) .* ((I_mat ./ k_matrix) .- i_k_mean).^2 .* μ .* x_opt))
    tobin_q = 1/abs(M - B) * sum((k_matrix .> 0.0) .* (p_val ./ k_matrix) .* μ .* x_opt)
    cash_flow_mean = 1/abs(M - B) * sum((k_matrix .> 0.0) .* (Π_mat ./ k_matrix) .* μ .* x_opt)
    cash_flow_std = sqrt(1/abs(M - B) * sum((k_matrix .> 0) .* ((Π_mat ./ k_matrix) .- cash_flow_mean).^2 .* μ .* x_opt))
    neg_i_frac = sum((I_mat .< 0) .* μ .* x_opt)

    # Generate tables
    table_2 = DataFrame(
        :Moments => ["Investment Share", 
        "Financial Cost Share",
        "Financial Cost to Total Cost",
        "Floatation Cost to Financial Cost"]
        ,:Value => round.([ I/Y , Λ/Y,  Λ/(Λ + production_cost), floatation_cost/Λ], digits = 4) )

    if save_tables
            table_2_latex = pretty_table(String, table_2, backend=:latex, alignment=:l, wrap_table=false, label="table_2")
        open("./PS5/document/table_2.tex", "w") do file
            write(file, table_2_latex)
        end
    end

    table_3 = DataFrame(:Moments => ["Average Size"
                ,"Investment Rate (mean)"
                ,"Investment Rate (st. dev)"
                ,"Tobin’s Q"
                ,"Cash Flow (mean)"
                ,"Cash Flow (st. dev)"
                ,"Frac. Negative Investment"],
            :Value => round.([size_mean, i_k_mean, i_k_std, tobin_q, cash_flow_mean, cash_flow_std, neg_i_frac], digits = 3))

    if save_tables 
        table_3_latex = pretty_table(String, table_3, backend=:latex, alignment=:l, wrap_table=false, label="table_2")
        open("./PS5/document/table_3.tex", "w") do file
            write(file, table_3_latex)
        end
    end

    ### Breakdown by divident/equity
    breakdown_constrained = Dict()
    breakdown_constrained[1] = sum((d_pf .< -0.1) .* μ)/sum(μ)
    breakdown_constrained[2] = sum((d_pf .> -0.3) .* (d_pf .< 0.32) .* μ)/sum(μ)
    breakdown_constrained[3] = sum((d_pf .>  0.32) .* μ)/sum(μ)


    table_dist = DataFrame(:Type => ["Externally Financed", "Constrained", "Unconstrained"],
            :Fraction => round.([breakdown_constrained[1],breakdown_constrained[2],breakdown_constrained[3]], digits = 3))
            
    
    if save_tables
        table_dist_latex = pretty_table(String, table_dist, backend=:latex, alignment=:l, wrap_table=false, label="table_dist")
        open("./PS5/document/table_dist.tex", "w") do file
            write(file, table_dist_latex)
        end
    end

    # Save results that we need for the next part
    res.CDF = CDF
    res.I_mat = I_mat
    res.Π_mat = Π_mat
end

# Similation and Regressions
function simulate_model(prim::Primitives, res::Results)

    @unpack ρ, σ, nZ, k_grid = prim
    @unpack z_vals, CDF, I_mat, Π_mat, x_opt, k_opt, p_val= res

    mc = tauchen(nZ, ρ, σ, 0, 4)
    T = 12 # 12 years we will throw away the first two
    N = 1200 # Number of firms
    
    # Simulating the shocks proces for each firm
    indices = [simulate_indices(mc, T ) for i in 1:N]
    indices = hcat(indices...)'
    
    # Initialize simulation objects
    z = zeros(N, T)
    k = zeros(N, T)
    x = zeros(N, T)
    v = zeros(N, T)
    i = zeros(N, T)
    Π = zeros(N, T)
    firm = zeros(N, T)
    year = zeros(N, T)
    
    
    for j in 1:N
        # draws initial productivity
        i_z = indices[j, 1]
        z[j, 1] = z_vals[i_z]
    
        # firm starts at a random level of capital based on the stationary distribution conditional on their initial productivity draw
        i_k = sum(CDF[i_z, :] .< rand())
        k[j, 1] = k_grid[i_k]
    
        # determines franchise value and capital for next period
        v[j, 1] = p_val[i_z, i_k]
        i[j, 1] = I_mat[i_z, i_k]
        Π[j, 1] = Π_mat[i_z, i_k]
        x[j, 1] = x_opt[i_z, i_k]
        k[j, 2] = k_opt[i_z, i_k]
        
        for t = 2:T
            # entrant know capital level and previous productivity level
            i_k = argmax(k[j, t] .== k_grid)
    
            # makes exit decision
            x[j, t] = x_opt[i_z, i_k]
            if x[j, t] == 0.0
                break
            end
            
            # draws next productivity
            i_z = indices[j, t]
            z[j, t] = z_vals[i_z]
    
            # determines franchise value
            v[j, t] = p_val[i_z, i_k]
    
            # Determines cash flow and investment 
            Π[j, t] = Π_mat[i_z, i_k]
            i[j, t] = I_mat[i_z, i_k]
            
            # Chooses capital for next period
            if (t < T)
                k[j, t+1] = k_opt[i_z, i_k]
            end
        end
    end
    
    z = z[:, 3:T]
    k = k[:, 3:T]
    x = x[:, 3:T]
    v = v[:, 3:T]
    i = i[:, 3:T]
    Π = Π[:, 3:T]
    firm = firm[:, 3:T]
    year = year[:, 3:T] .- 2
    
    # Estimate Regression
    x = x[:]
    i = i[:]
    k_lag = vcat(zeros(1, T-2), k[2:end ,:])[:]
    Π_lag = vcat(zeros(1, T-2), Π[2:end ,:])[:]
    v_lag = vcat(zeros(1, T-2), v[2:end ,:])[:]

    df = DataFrame(hcat(x, i./k_lag, v_lag./k_lag, Π_lag./k_lag), [:continuation, :investment, :tobinq, :cf])
    df.firm = reshape((1:N) * fill(1, T-2)', N *(T-2), 1)[:,1]
    df.year = reshape(fill(1, N) * (1:T-2)', N *(T-2), 1)[:,1]
    
    reshape((1:N) * fill(1, T-2)', N *(T-2), 1)[:,1]
    
    filter!(:continuation => x -> x == 1.0, df)
    filter!(:investment => x -> isfinite(x) & !isnan(x), df)
    filter!(:tobinq => x -> isfinite(x) & !isnan(x), df)
    filter!(:cf => x -> isfinite(x) & !isnan(x), df)
    println("# of unique firms: ", length(unique(df.firm)))
    
    return reg(df, @formula(investment ~ tobinq + cf + fe(year) + fe(firm)), Vcov.cluster(:firm))
end

function make_plots(prim::Primitives, res::Results; which="All")

    @unpack nZ, k_grid, Π, ι = prim
    @unpack x_opt, k_opt, z_vals, w, μ = res

	prod_levs = [1, Int(round(median(1:nZ))), nZ]
	labels = [L"z=z_l", L"z=z_m", L"z=z_h"]
	cols = [3, 1, 6]
	p_x_opt = plot(title="Exit", size = (700,700))
	p_div = plot(k_grid, zeros(size(k_grid)), label="",title="Dividend", c=:black, legend=:topleft, size = (700,700))
	p_k_opt = plot(k_grid, k_grid, label=L"k'=k", title="Capital", legend = :topleft, c=:black, size = (700,700))
	p_inv = plot(k_grid, zeros(size(k_grid)), label="",title="Investment", legend = :bottomleft, color=:black, size = (700,700))
	for i ∈ 1:3
		profit = Π.(k_grid, res.z_vals[prod_levs[i]], res.w)
		investment = ι.(k_grid, res.k_opt[prod_levs[i], :])
		dividend = profit - investment
		exit = 1 .- res.x_opt[prod_levs[i], :]
		
		plot!(p_x_opt, k_grid, exit, label=labels[i],c=cols[i])
		plot!(p_div, k_grid, dividend, label=labels[i], c=cols[i])
		plot!(p_k_opt, k_grid, res.k_opt[prod_levs[i], :],label=labels[i], c=cols[i])
		plot!(p_inv, k_grid, investment, label=labels[i],c=cols[i])
	end

	if w_plot == 1
		plot(p_k_opt)
	elseif w_plot == 2
		plot(p_x_opt)
	elseif w_plot == 3
		plot(p_div)
	elseif w_plot == 4
		plot(p_inv)
	else
		p_dec = plot( p_k_opt, p_x_opt, p_div, p_inv)
	end
	

    p_cdf = plot(title="CDF", size = (700, 700))

    for i ∈ 1:3
        μ = res.μ[prod_levs[i], :]
        cdf = cumsum( μ ) / sum(μ)
        plot!(p_cdf, k_grid, cdf, label=labels[i], c=cols[i])
    end
    return p_dec, p_cdf
    
end