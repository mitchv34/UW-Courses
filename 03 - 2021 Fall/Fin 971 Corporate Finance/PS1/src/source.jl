using DataFrames, GLM, TexTables, XLSX, FixedEffectModels, LaTeXStrings
using Plots

theme(:bright) 
default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

# path to data
path = "PS1/data/FHP_Illistration.xls"

# Reading data from file
data = DataFrame(XLSX.readtable(path, "Sheet1")...)

# Removing missing values 
dropmissing!(data)
# Removing NAN values 
filter!(:inv_rate => !isnan, data)
# Making sure data type is Float in the columns we need
cols = [:ppegt, :inv_rate, :cash_flow ,:tobin_q_lag]
data[!, cols] = Float64.(data[!, cols])
# Creating groups based on bins
gdata = DataFrames.groupby(data, :bin)
# Getting names of firms in each group
firm_names = [gd[1, :conml] for gd ∈ gdata]
# Selecting columns of interest for summary
# Present summary statistics
summary_table = summarize_by(data, :conml, cols)

write_tex("PS1/Document/tables/sum_table.tex", summary_table)

 Plots.plot(gdata[1][!, :fyear], gdata[1][!, :ppegt], label = firm_names[1], legend=:topleft)
Plots.plot!(gdata[1][!, :fyear], gdata[2][!, :ppegt], label = firm_names[2])
Plots.plot!(gdata[1][!, :fyear], gdata[3][!, :ppegt], label = firm_names[3])

Plots.savefig("PS1/Document/figures/fig_1.pdf")

plots = []
for i ∈ [1,2,3]
	p1 = plot(gdata[i][!, :fyear], gdata[1][!, :inv_rate],label="", title =L"I/K", linewidth=2)
	p2 = plot(gdata[i][!, :fyear], gdata[i][!, :tobin_q_lag],label="", title =L"Q", linewidth=2)
	p3 = plot(gdata[i][!, :fyear], gdata[i][!, :cash_flow],label="", title =L"CF", linewidth=2)
	push!(plots, plot(p1, p2, p3, layout=(1,3), size=(950, 300)))
	savefig("PS1/Document/figures/fig_2_$(firm_names[i]).pdf")
end


# First we estimate the regression model when β₂ = 0

labels = [ "tobin_q_lag" => "\$Q_{t−1}\$", "cash_flow" => "\$CF_{t}\$", "inv_rate" => "\$i_{t}\$"]
begin
	regs1 = [fit(LinearModel, @formula(inv_rate ~ tobin_q_lag), dat) for dat in gdata]
	table_cols1 = [TableCol(firm_names[i], regs1[i]) for i ∈ 1:3]
	table_first_reg = regtable(table_cols1..., labels=labels)
	savefig("PS1/Document/figures/fig_2_$().pdf")
end

write_tex("PS1/Document/tables/reg_table_1.tex", table_first_reg)

begin
	regs2 = [fit(LinearModel, @formula(
										inv_rate ~ tobin_q_lag + cash_flow), dat)
			for dat in gdata]
	table_cols2 = [TableCol(firm_names[i], regs2[i]) for i ∈ 1:3]
	table_second_reg =regtable(table_cols2...)
end

write_tex("PS1/Document/tables/reg_table_2.tex", table_second_reg)