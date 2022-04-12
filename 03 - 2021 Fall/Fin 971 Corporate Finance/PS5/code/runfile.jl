using  RegressionTables, PrettyTables
include("model.jl")

prim, res = initialize(0.08, 0.28)
solve_wage(prim, res)
solve_mass_entrants(prim, res)

#### Get Aggregates and cross-sectional moments
get_agg_results(prim, res)

### Generate Plots
w_plot = "ALL"
p_dec, p_cdf = make_plots(prim, res; which=w_plot)

savefig(p_dec, "./PS5/document/pol_funcs.pdf")
savefig(p_cdf, "./PS5/document/cdf.pdf")


#### Simulate model
reg_friction = simulate_model(prim, res)

### Re-do the whole thing in a model without frictions
_, res_no_f = initialize(0.0, 0.0)
solve_wage(prim, res_no_f)
solve_mass_entrants(prim, res_no_f)
get_agg_results(prim, res_no_f)
reg_no_friction = simulate_model(prim, res_no_f)

labels = Dict("tobinq" => "\$Q_{t-1}\$", 
                "cf" => "\$\\pi_{t-1} / k_{t-1}\$",
                "year" => "FE Year",
                "firm" => "FE Firm" ,
                "investment" => "\$i_{t}$")
number_regressions_decoration(s) = (s == 1) ? "Finantial Frictions ($s)" : "No Finantial Frictions // ($s)"
regtable(reg_friction, reg_no_friction, labels  = labels, number_regressions_decoration = number_regressions_decoration, 
            renderSettings = latexOutput("PS5/document/regression_table.tex"))