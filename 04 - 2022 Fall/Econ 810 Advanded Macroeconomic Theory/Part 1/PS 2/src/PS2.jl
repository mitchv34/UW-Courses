using DataFrames, CSV, Statistics, Distributions, Plots, StatsPlots
using StatsModels, GLM, FixedEffectModels, CategoricalArrays

begin
    theme(:vibrant); # :dark, :light, :plain, :grid, :tufte, :presentation, :none
    default(fontfamily="Computer Modern", framestyle=:box); # LaTex-style
    gr(size = (800, 600)); # default plot size
end


# Pat 1 (Data)
## PSID Data

PSID_data = CSV.read("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 2/data/income_diff.csv", DataFrame)

μ_ΔIncome = mean(PSID_data.delta_income)

## Simulation
N = 1000 # number of simulations
T = 11 # Total years
μ_Income = 30000 # mean of initial Income
init_income = filter(row -> row.Year == 1979, PSID_data).income
σ_ε = 1000 
σ_ν = 1000
dist_ε = Normal(0, σ_ε)
dist_ν = Normal(0, σ_ν)
dist_ζ = Normal(0, σ_ν)

### Generate simulated sample
initial_sample = μ_Income*ones(N) + rand(dist_ε, N)

data = zeros(N, T) # Preallocate space for data
data[:,1] = initial_sample # Initialize first column

# Simulate 5 years of full employment
for i ∈ 2:5
    data[:,i] = data[:,i-1] + rand(dist_ν, N)
end

# Simulate a random unemployment shock 

unemployed = sample(1:1000, 500, replace = false)
# Simulate unemployed earnings shock
data[unemployed,6] = data[unemployed,5] .- 9000 + rand(dist_ζ, Int(N/2))
# Simulate employed earnings
employed = [i for i ∈ 1:1000 if ~(i ∈ unemployed)]
data[employed, 6]  = data[employed, 5] + rand(dist_ν, Int(N/2))

# Simulate years 7 to 11
for i ∈ 7:11
    data[:,i] = data[:,i-1] + rand(dist_ν, N)
end

plot( data[unemployed, :]', color=1, legend=false , alpha=0.05)
plot!( data[employed, :]', color=2, legend=false, alpha=0.05) 

plot!( mean(data[unemployed, :], dims=1)', color=1, lw=3)
plot!(mean(data[employed, :], dims=1)', color=2, lw=3)

# Store data in DataFrame
begin
    income = data[:]
    year = repeat([i for i in 1:T], N)
    id = repeat([i for i in 1:N], 1,T)'[:]

    
    df = DataFrame([:id => id, :year => year, :income => income])
    years_from_layoff = df.year .- 6
    df.years_from_layoff = years_from_layoff
    df[employed, :years_from_layoff] .= Inf
    
    df[!, :years_from_layoff] = categorical(df[!, :years_from_layoff])

end

@df df[df.year .== 11, :] density(:income, legend = false, lw= 3)

df.years_from_layoff = [(y != Inf) ? y : missing for y in df.years_from_layoff]

# Estimate regression
lag_ols_reg = lm(@formula(income ~ year + years_from_layoff), df)



CSV.write("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 2/data/income_diff_sim.csv", df)
