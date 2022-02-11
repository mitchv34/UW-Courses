using DataFrames, CSV, Statistics, Distributions, Plots, StatsPlots
using StatsModels, GLM, FixedEffectModels, CategoricalArrays

begin
    theme(:juno); # :dark, :light, :plain, :grid, :tufte, :presentation, :none
    default(fontfamily="Computer Modern", framestyle=:box,); # LaTex-style
    gr(size = (800, 600)); # default plot size
    Plots.scalefontsizes(1.75)
end


# Pat 1 (Data)
## PSID Data

PSID_data = CSV.read("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 2/data/income_diff.csv", DataFrame)

μ_ΔIncome = mean(PSID_data.delta_income)

## Simulation
### Simulation parameters
T = 11 # number of periods
N_agents = 1000 # number of agents
N_unemployed = 500 # number of unemployed agents
N_employed = N_agents - N_unemployed # number of employed agents
# idiosyncratic component
σ_ε = 1000.0
μ_ε = 0.0
# time component
γ_1 = 30000.0
γ_t = 1000.0
# Unemployment shock
unemp_t = 5
unemp_shock = -9000.0

### Simulation
ε = rand(Normal(μ_ε, σ_ε), N_agents, T)
γ = γ_1 * ones(N_agents, T) + γ_t * repeat(0:10, 1, N_agents)'

# Draw sample of unemployed agents
unemployed = sample(1:N_agents, N_unemployed, replace = false)
employed = setdiff(1:N_agents, unemployed)
# Create dummy variables for employed and unemployed
D = [zeros(N_agents, unemp_t) ones(N_agents, T - unemp_t)]
D[employed, :] .= 0

data = γ + ε + unemp_shock * D

begin
    plot( data[unemployed, :]', color=1, legend=false , alpha=0.05)
    plot!( data[employed, :]', color=2, legend=false, alpha=0.05) 
    xticks!(1:11)

    plot!( mean(data[unemployed, :], dims=1)', color=1, lw=3)
    plot!(mean(data[employed, :], dims=1)', color=2, lw=3)
end


# Store data in DataFrame
begin
    income = data[:]
    year = repeat([i for i in 1:T], N)
    id = repeat([i for i in 1:N], 1,T)'[:]

    
    df = DataFrame([:id => id, :year => year, :income => income])
    years_from_layoff = df.year .- 5
    df.years_from_layoff = years_from_layoff
    df[employed, :years_from_layoff] .= Inf
    
    df.years_from_layoff = [(y  < 6 ) ? y : missing for y in df.years_from_layoff]
    df[!, :years_from_layoff] = categorical(df[!, :years_from_layoff])
    
    # df.years_from_layoff = [(y != 6) ? y : missing for y in df.years_from_layoff]
end

levels(df.years_from_layoff)

@df df[df.year .== 11, :] density(:income, legend = false, lw= 3)


# Estimate regression
## Manual approach (not recommended I do not like it)
## Create Dummy variables
for i in 1:T
    println(i-5)
end

## Using R
using RCall
reg = reval("lm(income ~ year + years_from_layoff) data = $df")

lag_ols_reg = lm(@formula(income  ~  id + years_from_layoff), df,
                        contrasts = Dict(:years_from_layoff => DummyCoding(),:id => EffectsCoding()))

lag_ols_reg

fit(df,  @formula( income ~ years_from_layoff + year))



reg(df, @formula(income ~ fe(id) + fe(year) + years_from_layoff))


CSV.write("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 2/data/income_diff_sim.csv", df)
