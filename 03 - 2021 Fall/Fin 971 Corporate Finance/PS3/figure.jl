using Plots, LaTeXStrings

theme(:vibrant) 
default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

x = 0:0.001:1
A_bar = 0.25
A_hat = 0.75
function f(x)
    if (x <= A_bar)
        return 0
    elseif (x > A_bar && x <= A_hat)
        return -(x - A_hat)
    else
        return 0
    end
end

y = f.(x)

begin
    plot(x, y,  xticks = (0:0.2:1, []), yticks = (0:0.2:1, []), ylims = [0, .6], color=:black, label="" )
    plot!( [A_bar, A_bar], [0, .6], label = L"\bar{A}", linestyle = :dash, c = 1, linewidth = 1.5)
    plot!( [A_hat, A_hat], [0, .6], label = L"\hat{A}", linestyle = :dash, c = 6, linewidth = 1.5)
    savefig("./PS3/Document/figures/exercise_2_1.pdf")
end

pwd()