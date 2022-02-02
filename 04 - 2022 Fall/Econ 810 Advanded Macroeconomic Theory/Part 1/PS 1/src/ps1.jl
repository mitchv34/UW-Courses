using DataFrames, CSV, FixedEffectModels

include("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/src/tauchen.jl")

data = CSV.read("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/data/data_1.csv", DataFrame)

data.income  = log.(data.Income)

reg(data, @formula(income ~ fe(Age) + fe(Year)))

ρ = 0.97

tauchenMethod(0, 1.0, ρ, 5, q=3) 