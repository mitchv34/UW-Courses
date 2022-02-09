# Load libraries
using DataFrames, CSV, Plots, QuantEcon, CategoricalArrays, GLM, Statistics, StatsPlots

# Plot style
begin
    theme(:juno); # :dark, :light, :plain, :grid, :tufte, :presentation, :none
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
age_profile = coef(ols_reg)[2:36];
cohort_profile = coef(ols_reg)[37:end];

# Make plots
scatter(levels(data.age)[2:end], age_profile, xlab = "Age", ylab = "Coefficient", legend=false);
scatter(levels(data.cohort)[2:end], cohort_profile, xlab = "Cohort", ylab = "Coefficient", legend=false);

# Estimate shock variances
y = residuals(ols_reg);
ρ = 0.97;
Δy_t = y[2:end] - ρ*y[1:end-1];
σ_ε = -(1/ρ)*cov(Δy_t[1:end-1], Δy_t[2:end]);
σ_ζ =(1/ρ)*cov(Δy_t[2:end-1],  ρ^2*Δy_t[1:end-2] + ρ*Δy_t[2:end-1] + Δy_t[3:end]);

# Calculate the age-income profiles
data.κ = data.log_income - y;
κ = [ mean(data[data.age .== a, :κ]) for a ∈ 25:(25+34) ];
plot(levels(data.age)[2:end], exp.(κ));

@df data violin(string.(:year), :income, linewidth=0)
@df data boxplot!(string.(:year), :income, fillalpha=0.4, linewidth=2)

@df data violin(string.(:age), :income, linewidth=0)
@df data boxplot!(string.(:age), :income, fillalpha=0.4, linewidth=2)

@df data scatter(:year, :income)

include("model.jl")
prim, res = initialize();
Bellman_op(prim, res)

plot(res.value_fun[:, :, 2, end-1], legend = false)
plot(prim.A_grid, res.pol_fun[1:end, 1, 1, end-1], legend = false)


sim_res = sim_data(2000, 0, prim, res)

begin
    plot(var( C_agents, dims=1)', label="Consumption", lw = 2)
    plot!(var( A_agents, dims=1)', label="Assets", lw=2)
    plot!(var( Y_agents, dims=1)', label="Income", lw=2)
end