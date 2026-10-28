using LinearAlgebra
using DifferentialEquations
using Printf
using Random
using LaTeXStrings
using Plots
using StatsBase
using Graphs

Random.seed!(42)

const N  = 6
const T  = 10.0
const dt = 0.01
const α  = 1.0
const nt = Int(round(T / dt))
const ntp1 = nt + 1
const ts = collect(range(0.0, T; length = ntp1))
const wc = 0.0005
const we = 1.0

const t_prec_start = 2.0
const t_prec_end   = 10.0
@inline precision_window(t) = (t_prec_start <= t <= t_prec_end) ? we*1.0 : 0.0
const w_prec_grid = [precision_window(t) for t in ts]


#=
# Build all edge coordinates once=#

all_i = Vector{Int}(undef, N*N-N)
all_j = Vector{Int}(undef, N*N-N)
p = 1
for i in 1:N, j in 1:N
    global p
    if(i!=j)
        all_i[p] = i
        all_j[p] = j
        p += 1
    end
end

# ai
edge_ind = 1:(N*N-N)#sample(1:(N*N-N), 200, replace=false)
# bi
#edge_ind = sample(1:(N*N-N), 245, replace=false)
# ci
#edge_ind = sample(1:(N*N-N), 196, replace=false)
#edge_ind = sample(1:(N*N-N), 195, replace=false)

const edge_i = all_i[edge_ind]
const edge_j = all_j[edge_ind]

Adja = zeros(N,N)
for i in 1:length(edge_i)
    Adja[edge_i[i],edge_j[i]]=1.0
end
#iii
#r_star = [randn(Int64(N/2)).+2.0; randn(Int64(N/2)).-2.0]
#ii
#r_star = 0.05*randn(N)
#i

#Random.seed!(90920000)
#Random.seed!(10920000)
Random.seed!(42)

r_star = 1.0*randn(N)

r0 = copy(r_star)
const t_state= (1*rand([1,1.5,2],N))

function r_tgt_func(t)
   # s = (cos(2π * t * 3.0 / T) * cos(2π * t * 4.0 / T + 0.1))
    
    
    s = -1.0.+(cos(2π * t * 3.0 / T) * cos(2π * t * 4.0 / T + 0.1))
    s2 = 1.0.+(cos(2π * t * 2.0 / T) * cos(2π * t * 3.0 / T + 0.1))

    #return fill(s, N)
    return [s.*t_state[1:Int64(N/2)];s2.*t_state[Int64(N/2)+1:end]]
end
const target_vals = hcat((r_tgt_func(t) for t in ts)...)

@inline trapz_weight(k) = (k == 1 || k == ntp1) ? 0.5 : 1.0

@inline function time_index_and_weight(t)
    x = t / dt
    idx = floor(Int, x) + 1
    if idx < 1
        return 1, 0.0
    elseif idx >= ntp1
        return nt, 1.0
    else
        return idx, x - (idx - 1)
    end
end

function get_U_t!(Ut, t, U)
    idx, w = time_index_and_weight(t)
    @inbounds @views for j in 1:N, i in 1:N
        Ut[i, j] = (1 - w) * U[i, j, idx] + w * U[i, j, idx + 1]
        if(i==j)
            Ut[i, i] = 0.0
        end
    end
    return idx
end

mutable struct EvalWorkspace
    Ut_fwd::Matrix{Float64}
    Ut_adj::Matrix{Float64}
    A_fwd::Matrix{Float64}
    A_adj::Matrix{Float64}
    R::Matrix{Float64}
    Λ::Matrix{Float64}
    grad::Array{Float64,3}
end

function EvalWorkspace(N, nt)
    EvalWorkspace(
        zeros(N, N),
        zeros(N, N),
        zeros(N, N),
        zeros(N, N),
        zeros(N, nt + 1),
        zeros(N, nt + 1),
        zeros(N, N, nt + 1),
    )
end

function evaluate_objective_and_gradient!(ws::EvalWorkspace, U, r0, r_star)
    Ut_fwd = ws.Ut_fwd
    Ut_adj = ws.Ut_adj
    A_fwd  = ws.A_fwd
    A_adj  = ws.A_adj
    R      = ws.R
    Λ      = ws.Λ
    grad   = ws.grad

    fill!(grad, 0.0)

    function forward_ode!(dr, r, p, t)
        get_U_t!(Ut_fwd, t, U)
        copyto!(A_fwd, Ut_fwd)
        @inbounds for i in 1:N
            A_fwd[i, i] -= α
        end
        mul!(dr, A_fwd, r)
        @inbounds for i in 1:N
            dr[i] += α * r_star[i]
        end
        return nothing
    end

    prob_r = ODEProblem(forward_ode!, r0, (0.0, T))
    sol_r = solve(prob_r, Tsit5(); reltol=1e-4, abstol=1e-5,
                  saveat=ts, save_everystep=false, dense=false)
    R .= Array(sol_r)

    function adjoint_ode_reversed!(dλ, λ, p, t)
        idx = get_U_t!(Ut_adj, t, U)

        @inbounds for j in 1:N, i in 1:N
            A_adj[i, j] = -Ut_adj[j, i]
        end
        @inbounds for i in 1:N
            A_adj[i, i] += α
        end
        mul!(dλ, A_adj, λ)

        wp = precision_window(t)
        @inbounds for i in 1:N
            dλ[i] += wp * (R[i, idx] - target_vals[i, idx])/N
        end
        return nothing
    end

    prob_λ = ODEProblem(adjoint_ode_reversed!, zeros(N), (T, 0.0))
    sol_λ = solve(prob_λ, Tsit5(); reltol=1e-4, abstol=1e-5,
                  saveat=ts, save_everystep=false, dense=false)
    Λ .= reverse(Array(sol_λ), dims=2)

    Fu = 0.0
    Fe = 0.0
    @inbounds for k in 1:ntp1
        wk = trapz_weight(k)
        Fu += wk * 0.5 * sum(abs2, @view U[:, :, k])

        s = 0.0
        for i in 1:N
            d = R[i, k] - target_vals[i, k]
            s += d*d
        end
        Fe += wk * 0.5 * w_prec_grid[k] * s /N
    end
    Fu *= dt
    Fe *= dt
    C = wc * Fu + we * Fe

    @inbounds for k in 1:ntp1
        for e in eachindex(edge_i)
            i = edge_i[e]
            j = edge_j[e]
            grad[i, j, k] = wc * U[i, j, k] - Λ[i, k] * R[j, k]
        end
    end

    return C, wc * Fu, we * Fe, grad, R
end


# ─── Adam Optimizer ─────────────────────────────────────────────────────────
epochs = 35000
#η      = 0.00001
η      = 0.001

β1     = 0.9
β2     = 0.999
ϵ      = 1e-8

#U = 0.0*ones(N, N, nt + 1)
#
U = 0.1*ones(N, N, nt + 1)
#U = -0.02*rand(N, N, nt + 1)
#U[1:3,1:3,:] = -0.1*U[1:3,1:3,:]

m = zeros(N, N, nt + 1)
v = zeros(N, N, nt + 1)

cost_hist = Vector{Float64}(undef, epochs)
Fu_hist   = Vector{Float64}(undef, epochs)
Fe_hist   = Vector{Float64}(undef, epochs)
grad_hist = Vector{Float64}(undef, epochs)
rate_hist = Vector{Matrix{Float64}}(undef, epochs)

ws = EvalWorkspace(N, nt)

println("Starting Adam Optimization ...")
println("------------------------------------------------------------------------")
@printf("%-6s | %-12s | %-12s | %-12s | %-12s\n",
        "Epoch", "Total Cost", "Fu (Ctrl)", "Fe (Prec)", "||Grad||")
println("------------------------------------------------------------------------")
ep = 0
for epoch in 1:epochs
    C, Fu, Fe, grad, R = evaluate_objective_and_gradient!(ws, U, r0, r_star)
    gnorm = norm(grad,Inf)
    if(gnorm<0.001)
    #if(gnorm<0.0000001)

         @printf("%-6d | %-12.6f | %-12.6f | %-12.6f | %-12.6f\n",
                epoch, C, Fu, Fe, gnorm)
        ep = epoch 
        cost_hist[end] = C
        Fu_hist[end]   = Fu
        Fe_hist[end]   = Fe
        grad_hist[end] = gnorm
        rate_hist[end] = copy(R)
        break;
    end
    cost_hist[epoch] = C
    Fu_hist[epoch]   = Fu
    Fe_hist[epoch]   = Fe
    grad_hist[epoch] = gnorm
    rate_hist[epoch] = copy(R)

    if epoch % 50 == 0 || epoch == 1
        @printf("%-6d | %-12.6f | %-12.6f | %-12.6f | %-12.6f\n",
                epoch, C, Fu, Fe, gnorm)
    end

    @. m = β1 * m + (1.0 - β1) * grad
    @. v = β2 * v + (1.0 - β2) * grad^2

    β1corr = 1.0 - β1^epoch
    β2corr = 1.0 - β2^epoch
    @. U = U - η * (m / β1corr) / (sqrt(v / β2corr) + ϵ)


end

println("------------------------------------------------------------------------")
println("Optimization Complete.")
println("Norm of U at t=0 : ", round(norm(U[:, :, 1]), digits = 4))
println("Norm of U at t=T : ", round(norm(U[:, :, end]), digits = 4))

C_opt, Fu_opt, Fe_opt, grad_opt, R_opt = evaluate_objective_and_gradient!(ws, U, r0, r_star)

println("Final recomputed cost      : ", round(C_opt, digits = 6))
println("Final control cost Fu      : ", round(Fu_opt, digits = 6))
println("Final precision cost Fe    : ", round(Fe_opt, digits = 6))
println("Final gradient norm        : ", round(norm(grad_opt,Inf), digits = 6))

# ─── Plots ───────────────────────────────────────────────────────────
gf = 18
tf = 18
pl_traj = plot(
    ts, R_opt[1, :],
    xlabel = "time [a.u.]",
    ylabel = L"x_i(t)",
    guidefontsize=18,
    label = L"x_1",
    linewidth = 2,
    tickfont=font(tf,"Computer Modern"), guidefont=font(gf,"Computer Modern")#,
   # title = "Optimal state trajectories"
)

for i in 2:N
    plot!(pl_traj, ts, R_opt[i, :], label = L"x_" * "$(i)", linewidth = 2, legend=false)
end

for i in 1:N
    plot!(pl_traj, ts, target_vals[i, :], label = false, linestyle = :dash, linewidth = 1.5, color = "black", alpha=0.1)
end

vspan!(pl_traj, [t_prec_start, t_prec_end], color = :gray, alpha = 0.15, label = "precision window")
title!(L"\mathcal{C}_e"*": $(round(Fe_opt,digits=5)); "*L"\mathcal{C}_u"*": $(round(Fu_opt,digits=4))",titlefont = font(24, "Computer Modern"))

epoch_axis = 1:epochs
pl_cost = plot(epoch_axis, cost_hist, xlabel = "epoch", ylabel = "cost", label = "Total cost", linewidth = 2,
               title = "Cost evolution during optimization")
plot!(pl_cost, epoch_axis, Fu_hist, label = "wc * Fu (weighted control)", linewidth = 2, linestyle = :dash)
plot!(pl_cost, epoch_axis, Fe_hist, label = "Fe (precision)", linewidth = 2, linestyle = :dot)

U_flat = reshape(U, N * N, nt + 1)
#connection_labels = ["U[$i,$j]" for i in 1:N for j in 1:N]
#pl_weights = heatmap(ts, 1:(N * N), U_flat, xlabel = "time [a.u.]", ylabel = "connection index",
#                     title = "Optimal network weights over time", colorbar_title = "weight")
#if N <= 8
#    plot!(pl_weights, yticks = (1:(N * N), connection_labels))
#end

pl_selw = plot(xlabel = "time [a.u.]",tickfont=font(tf,"Computer Modern"), guidefont=font(gf,"Computer Modern"), ylabel = L"U_{ij}(t)")#, title = "Selected optimal weights")
#for i in 1:min(N, 80), j in 1:min(N, 80)
for i in 1:length(edge_i)
    plot!(pl_selw, ts, vec(U[edge_i[i], edge_j[i], :]), legend = false,guidefontsize=18, linewidth = 2)

  #  if(i!=j)
  #      plot!(pl_selw, ts, vec(U[i, j, :]), legend = false, label = "U[$i,$j]",guidefontsize=18, linewidth = 2)
  #  end
end

savefig(pl_traj, "optimal_trajectories.png")
savefig(pl_cost, "cost_evolution.png")
#savefig(pl_weights, "optimal_weights_heatmap.png")
savefig(pl_selw, "selected_weights.png")
#pl_all = plot(pl_traj, pl_cost, pl_weights, pl_selw, layout = (4, 1), size = (950, 1600))
pl_all = plot(pl_traj,  pl_selw, layout = (2, 1), size = (950, 800),tickfontsize=12)
savefig(pl_all, "summary_plots.pdf")
savefig(pl_all, "summary_plots_ai.png")
savefig(pl_all, "summary_plots_ai.pdf")