# Load libraries
using DataFrames, CSV, Plots, QuantEcon, CategoricalArrays, GLM, Statistics, StatsPlots

# Plot style
begin
    theme(:vibrant); # :dark, :light, :plain, :grid, :tufte, :presentation, :none
    default(fontfamily="Computer Modern", framestyle=:box); # LaTex-style
    gr(size = (800, 600)); # default plot size
end

# Load data
data = CSV.read("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/data/data_1_alt.csv", DataFrame);
dropmissing!(data);
data.log_income  = log.(data.income);
data.age = CategoricalArray(data.age);
data.cohort = CategoricalArray(data.cohort);

data

p1= @df data violin(string.(:year), :income, linewidth=0)
@df data boxplot!(string.(:year), :income, fillalpha=0.4, linewidth=2)
savefig(p1, "04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/document/figures/data_less_outliers.pdf");

# Estimate regression
ols_reg = lm(@formula(log_income ~ age + cohort), data);
# Get age_profile and cohort_profile
age_profile = coef(ols_reg)[2:36]
cohort_profile = coef(ols_reg)[37:end];

# Make plots
scatter(levels(data.age)[2:end], age_profile, xlab = "Age", ylab = "Coefficient", legend=false);
scatter(levels(data.cohort)[2:end], cohort_profile, xlab = "Cohort", ylab = "Coefficient", legend=false);

# Calculate the age-income profiles
data.κ = data.log_income - y;
κ = [ mean(data[data.age .== a, :κ]) for a ∈ 25:(25+34) ];
plot(levels(data.age)[2:end], exp.(κ), xticks = levels(data.age)[2:3:end], xlab = "Age", ylab = "Income")
savefig("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/document/figures/age_income_profile.pdf")

# Estimate shock variances
y = residuals(ols_reg);
ρ = 0.97;
function estimate_variance(ρ::Float64, y::Array{Float64})
    Δy_t = y[2:end] - ρ*y[1:end-1];
    σ_ε = -(1/ρ)*cov(Δy_t[1:end-1], Δy_t[2:end])
    σ_ζ =(1/ρ)*cov(Δy_t[2:end-1],  ρ^2*Δy_t[1:end-2] + ρ*Δy_t[2:end-1] + Δy_t[3:end])
    return σ_ε, σ_ζ
end

σ_ε, σ_ζ = estimate_variance(ρ, y)

rhos = [0.90, 0.95, 0.97, 0.99];
vars = round.(hcat(collect.(estimate_variance.(rhos, Ref(y)))...), digits=3)

using LaTeXStrings, TexTables, Latexify
copy_to_clipboard(true)
latex_table = latexify(DataFrame([L"\rho"=>rhos, L"\sigma_\varepsilon^2"=>vars[1,:], L"\sigma_\zeta^2"=>vars[2,:]]), env = :table)



@df data violin(string.(:age), :income, linewidth=0)
@df data boxplot!(string.(:age), :income, fillalpha=0.4, linewidth=2)

@df data scatter(:year, :income)

include("model.jl")
prim, res = initialize();
Bellman_op(prim, res)

plot(res.value_fun[:, :, 2, end-1], legend = false)

sim_res = sim_data(5000, 0.0, prim, res)

# Plot average value of wealth by age
nAgents = 5000
agents = 1:nAgents
ages = 25:25+prim.T-1

@unpack C_sim, A_sim, Y_sim = sim_res;

begin
     plot(levels(data.age)[2:end],  mean( C_sim, dims=1)', c=1, label="Consumption", lw = 2)
    plot!(levels(data.age)[2:end],  mean( A_sim, dims=1)', c=2, label="Assets", lw=2)
    plot!(levels(data.age)[2:end],  mean( Y_sim, dims=1)', c=4, label="Income", lw=2)
    xticks!(levels(data.age)[2:3:end])
    savefig("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/document/figures/average_value_of_wealth_by_age.pdf")
end


data_sim = DataFrame([:agent_id => repeat(agents, prim.T),
                    :age => repeat(ages', nAgents)[:],
                        :consumption => C_sim[:],
                        :income => Y_sim[:],
                        :assets => A_sim[:]])


# Plot variance of consumption by age
plot(var( C_sim, dims=1)', c=1, lw = 2, legend=false)
savefig("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/document/figures/variance_of_consumption_by_age.pdf")
# @df data_sim boxplot(string.(:age), :consumption, fillalpha=0.4, linewidth=2)

# Insurance coefficients using the Blundell et al. (2008) method
# Pass through of transitory shock:
#   ψ = cov(Δc_t, Δy_t+1) / cov(Δy_t, Δy_t+1)
c_sim = mean(log.(C_sim), dims=1) 
y_sim = mean(log.(Y_sim), dims=1)  

Δy = y_sim[2:end] - y_sim[1:end-1]
Δc = c_sim[2:end] - c_sim[1:end-1]
ψ = cov(c_sim[1:end-1], y_sim[2:end]) / cov(y_sim[1:end-1], y_sim[2:end])

# Pass through of permanent shock: 
#   ϕ = cov(Δc_t, Δy_t-1 + Δy_t + Δy_t+1) / cov(Δy_t, Δy_t-1 + Δy_t + Δy_t+1)
ϕ = cov(c_sim[1:end-2], y_sim[1:end-2] + y_sim[2:end-1] + y_sim[3:end]) / cov(y_sim[1:end-2],y_sim[1:end-2] + y_sim[2:end-1] + y_sim[3:end])


minimum(C_sim, dims=1)

A_sim

for t in prim.T, i in 1:prim.nA, j in prim.nP, k in prim.nε
    y = res.Y[j,k,t]
    a = prim.A_grid[i]
    a_next = res.pol_fun[i,j,k,t]
    c = (1+prim.r)*a + y - a_next
    if c < 0
        print(c)
    end
end

@df data_sim scatter(:age, :assets)0