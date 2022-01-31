using StatFiles, DataFrames

df = DataFrame(load("04 - 2022 Fall/Econ 810 Advanded Macroeconomic Theory/Part 1/PS 1/data/pequiv_long.dta"))
dropmissing!(df)


df

filter(row -> row.x11104LL == 11, df)