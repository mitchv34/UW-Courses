using Parameters, Distributed, SharedArrays, ProgressBars,StatsBase# Distributions, 
include("tauchen.jl")

using Plots
begin
    theme(:juno); # :dark, :light, :plain, :grid, :tufte, :presentation, :none
    default(fontfamily="Computer Modern", framestyle=:box,); # LaTex-style
    gr(size = (800, 600)); # default plot size
    Plots.scalefontsizes(1.75)
end



# Primitive structure
@with_kw struct Primitives
# @with_kw struct 
    T::Int64                    = 360
    r::Float64                  = 0.04
    β::Float64                  = (1/(1+r))^(1/12)
    δ::Float64                  = 0.033
    b::Float64                  = 0.1

    h_min::Float64              = 1.0
    h_max::Float64              = 2.0
    Nh::Int64                   = 25
    Δ::Float64                  = 1/(Nh - 1)
    h_grid::Array{Float64,1}    = range(h_min, h_max, length=Nh)

    s_min::Float64              = 0.0
    s_max::Float64              = 1.0
    Ns::Int64                   = 41
    s_grid::Array{Float64,1}    = range(s_min, s_max, length=Nh)

    # Wage
    Nw::Int64                   = 41
    tauchenResult = tauchenMethod(0.5, 0.1, 0.0, 41) 
    w_grid::Array{Float64}    = tauchenResult[1]
    p_w::Array{Float64,1}       = tauchenResult[2][1,:]

    ψ_e::Float64                = 0.5
    ψ_u::Float64                = 0.25
end


mutable struct Results
    s_pol::SharedArray{Float64,2}
    w_pol::SharedArray{Float64,2}
    U_val::SharedArray{Float64,2}
    W_val::SharedArray{Float64,3}
end


function initialize()
    prim                        = Primitives()
    s_pol                       = SharedArray{Float64}(prim.Nh, prim.T)
    w_pol                       = SharedArray{Float64}(prim.Nh, prim.T)
    U_val                       = SharedArray{Float64}(prim.Nh, prim.T)
    W_val                       = SharedArray{Float64}(prim.Nw, prim.Nh, prim.T)

    # Initialize the last period policy and value functions
    s_pol[:, prim.T]           .= 0
    s_pol[:, prim.T]           .= 0
    w_pol[:, prim.T]           .= 0
    W_val[:,:, prim.T]         .= prim.w_grid*prim.h_grid'
    U_val[:, prim.T]           .= prim.b

    res                         = Results(s_pol, w_pol, U_val, W_val)
    return prim, res
end

## Bellman operator
function Bellman_op(prim::Primitives, res::Results)
    @unpack T, Nh, Ns, b, p_w, h_grid, w_grid, s_grid, β, δ, ψ_u, ψ_e = prim
    @unpack U_val, W_val, s_pol, w_pol = res
    # Loop backwards over time
    for t ∈ ProgressBar(T-1:-1:1)
        # Loop backwards over time
        for h_i ∈ 1:Nh
            h = h_grid[h_i]
            # Unemployed agents
            val_unemp_next = (1 - ψ_u)U_val[h_i, t+1] + ψ_u*U_val[max(h_i-1, 1), t+1]
            value_offer_no_change = p_w' * max.(W_val[:, h_i, t+1], U_val[h_i, t+1]) 
            value_offer_change = p_w' * max.(W_val[:, max(h_i-1, 1), t+1], U_val[max(h_i-1, 1), t+1])
            value_offer =  (1 - ψ_u)value_offer_no_change + ψ_u*value_offer_change
            Π_s = sqrt.(s_grid)
            exp_value = (1 .- Π_s) .* val_unemp_next .+ Π_s.*value_offer
            U = b .- 0.5*s_grid + β*exp_value
            # Update value function(s)
            U_val[h_i, t] = maximum(U)
            W_val[:, h_i, t] = h*w_grid.+(1-δ)*((1-ψ_e).*W_val[:,h_i,t+1].+ψ_e*W_val[:,min(h_i+1, Nh),t+1]).+δ.*((1-ψ_e).*U_val[h_i,t+1].+ ψ_e.*U_val[min(h_i+1, Nh),t+1])
            # Update policy function(s) 
            if  t == T-1
                println(U[1])
            end
            s_pol[h_i, t] = s_grid[argmax(U)]
            w_pol[h_i, t] = w_grid[findfirst(W_val[:, h_i, t+1] .- U_val[h_i, t+1] .> 0)]
        end # h unemployed
    end # T
end


# Struct to hold the results of the simulation
mutable struct Sim_results
    emp_status ::Array{Float64, 2} # 0: unemployed, 1: employed
    h_agents::Array{Float64, 2} # Human capital of agents
    # Y_agents::Array{Float64, 2} # Income
end


# Simulate the model
function sim_data(nAgents::Int64, prim::Primitives, res::Results)

    @unpack T, h_grid, Nh, ψ_u, ψ_e, w_grid, p_w, δ = prim
    @unpack s_pol, w_pol = res

    # Initialize the simulation resuts variablses
    emp_status = zeros(nAgents, T)
    h_agents = zeros(nAgents, T)

    # Initial human capital drawn from uniform distribution
    h_ind = rand(1:Nh, nAgents)
    h_agents[:,1] = h_grid[h_ind]

    for t ∈ 1:T-1
        # Loop over agents
        for agent ∈ 1:nAgents
            # List of things to do update for next period
            # [x] Update human capital for next period
            # [x] Update employment status for next period
            if emp_status[agent, t] == 0
                # Unemployed
                h_ind_next = max(1, sample([h_ind[agent], h_ind[agent] - 1], Weights([ψ_u, 1-ψ_u]), 1 )[1] ) # Update human capital
                s = s_pol[h_ind[agent], t] # Get searh effort
                offer = sample([0,1], Weights([1-sqrt(s), sqrt(s)]), 1 )[1] # Get offer or not
                if offer == 0
                    # No offer
                    emp_status_next = 0
                else
                    # Offer
                    w = sample(w_grid, Weights(p_w), 1)[1] # Get wage
                    emp_status_next = w_pol[h_ind[agent], t] >= w # Accept offer if wage above than reservation wage
                end
            else
                # Employed
                h_ind_next = min( Nh, sample([h_ind[agent], h_ind[agent] + 1], Weights([ψ_e, 1-ψ_e]), 1 )[1] ) # Update human capital
                emp_status_next = sample([0,1], Weights([δ, 1-δ]), 1)[1] # Employed workers loose their job with probability δ
            end
            h_agents[agent, t+1] = h_grid[h_ind_next] # Update human capital
            emp_status[agent, t+1] = emp_status_next # Update employment status
        end

    end

    return Sim_results(emp_status, h_agents)
end

prim, res = initialize()
Bellman_op(prim, res)
nAgents = 5000
@unpack emp_status,h_agents =  sim_data(nAgents, prim, res)

using DataFrames, StatsPlots
# Create DataFrame to hold the results
df = DataFrame([:time => repeat(1:nAgents, 1, prim.T)'[:],
                :id => repeat(1:prim.T, 1, nAgents)[:],
                :h_agents => h_agents[:],
                :emp_status => emp_status[:]
                ])

@df df[df.time .==  200, :] density(:h_agents, group = (:emp_status), legend = :topleft)
xlims!(1,2)

h_mean_unemp = [mean(data.h_agents) for data in groupby(df[df.emp_status .==  0, :], :time)]
h_mean_emp = [mean(data.h_agents) for data in groupby(df[df.emp_status .==  1, :], :time)]

plot(res.W_val[10,:, 1:50:end], legend=false)

plot(prim.h_grid, res.s_pol[:,1:end], legend=false)
plot(prim.h_grid, res.w_pol[:,1:end], legend=false)

df[df.time .==  1, :]