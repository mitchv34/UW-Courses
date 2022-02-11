# Load libraries
using DataFrames, CSV, Plots, QuantEcon, CategoricalArrays, GLM, Statistics, StatsPlots

# Plot style
begin
    theme(:vibrant); # :dark, :light, :plain, :grid, :tufte, :presentation, :none
    default(fontfamily="Computer Modern", framestyle=:box); # LaTex-style
    gr(size = (800, 600)); # default plot size
end

# Load data
data = CSV.read("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/data/data_1.csv", DataFrame);
data.log_income  = log.(data.income);
data.age = CategoricalArray(data.age);
data.cohort = CategoricalArray(data.cohort);

# Estimate regression
ols_reg = lm(@formula(log_income ~ age + cohort), data);

# Get age_profile and cohort_profile
age_profile = coef(ols_reg)[2:36]
cohort_profile = coef(ols_reg)[37:end];

# Make plots
scatter(levels(data.age)[2:end], age_profile, xlab = "Age", ylab = "Coefficient", legend=false);
scatter(levels(data.cohort)[2:end], cohort_profile, xlab = "Cohort", ylab = "Coefficient", legend=false);

# Estimate shock variances
y = residuals(ols_reg);
ρ = 0.97;
Δy_t = y[2:end] - ρ*y[1:end-1];
σ_ε = -(1/ρ)*cov(Δy_t[1:end-1], Δy_t[2:end])
σ_ζ =(1/ρ)*cov(Δy_t[2:end-1],  ρ^2*Δy_t[1:end-2] + ρ*Δy_t[2:end-1] + Δy_t[3:end])

# Calculate the age-income profiles
data.κ = data.log_income - y;
κ = [ mean(data[data.age .== a, :κ]) for a ∈ 25:(25+34) ];
plot(levels(data.age)[2:end], exp.(κ));

p1= @df data violin(string.(:year), :income, linewidth=0)
@df data boxplot!(string.(:year), :income, fillalpha=0.4, linewidth=2)

savefig(p1, "04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/document/figures/fig_1.pdf");

@df data violin(string.(:age), :income, linewidth=0)
@df data boxplot!(string.(:age), :income, fillalpha=0.4, linewidth=2)

@df data scatter(:year, :income)

include("model.jl")
prim, res = initialize();
Bellman_op(prim, res)

plot(res.value_fun[:, :, 2, end-1], legend = false)

sim_res = sim_data(2000, 0.0, prim, res)

# Plot average value of wealth by age
nAgents = 2000
agents = 1:nAgents
ages = 25:25+prim.T-1

C_agents[:, 1]

data_sim = DataFrame([:agent_id => repeat(agents, prim.T),
                    :age => repeat(ages', nAgents)[:],
                        :consumption => C_agents[:],
                        :income => Y_agents[:],
                        :assets => A_agents[:]])


begin
    @unpack C_agents, A_agents, Y_agents = sim_res;
    plot(mean( C_agents, dims=1)', c=1, label="Consumption", lw = 2)
    plot!(mean( A_agents, dims=1)', c=2, label="Assets", lw=2)
    plot!(mean( Y_agents, dims=1)', c=4, label="Income", lw=2)
end

# Plot variance of consumption by age
plot(var( C_agents, dims=1)', c=1, lw = 2, legend=false)

# Insurance coefficients using the Blundell et al. (2008) method
# ψ = 
Δc = log.(C_agents[:, 2:end]) - log.(C_agents[:, 1:end-1]); 

minimum(C_agents, dims=1)

A_agents

for t in prim.T, i in 1:prim.nA, j in prim.nP, k in prim.nε
    y = res.Y[j,k,t]
    a = prim.A_grid[i]
    a_next = res.pol_fun[i,j,k,t]
    c = (1+prim.r)*a + y - a_next
    if c < 0
        print(c)
    end
end

@df data_sim scatter(:age, :assets)