using DataFrames, CSV, FixedEffectModels, Plots, QuantEcon

include("model.jl")

# data = CSV.read("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/data/data_1.csv", DataFrame)

# data.income  = log.(data.Income)

# reg(data, @formula(income ~ fe(Age) + fe(Year)))

ρ = 0.97
σ_ζ = 0.02
σ_ε = 0.075

prim, res = initialize(ρ, σ_ζ, σ_ε)

ε, ε_mat = tauchenMethod(0.0, σ_ε, 0.0, 5)

stationary_distributions(MarkovChain(ε_mat))

sumε_mat