# Load libraries
using DataFrames, CSV, Plots, CategoricalArrays
using GLM, Statistics, StatsPlots, DelimitedFiles
using StatFiles

# Plot style
begin
    theme(:vibrant); # :dark, :light, :plain, :grid, :tufte, :presentation, :none
    default(fontfamily="Computer Modern", framestyle=:box); # LaTex-style
    gr(size = (800, 600)); # default plot size
end


@time df = DataFrame(load("/mnt/c/Users/mitch/Downloads/pequiv_long.dta"))




pre_proc_data = false

if pre_proc_data # TODO: Need to fix this, system does not find python script 
    run(`python "04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS5/pre_proc_data.py"`)
end

# find_κ = true

# if find_κ

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
y = residuals(ols_reg);
data.κ = data.log_income - y;
κ = [ mean(data[data.age .== a, :κ]) for a ∈ 25:(25+34) ];
plot(levels(data.age)[2:end], exp.(κ), xticks = levels(data.age)[2:3:end], xlab = "Age", ylab = "Income")

# end