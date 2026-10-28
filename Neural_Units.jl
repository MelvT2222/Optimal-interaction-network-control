using LinearAlgebra
using DifferentialEquations
using Printf
using Random
using LaTeXStrings
using Plots
using StatsBase
using Graphs
using FFTW
using Statistics

Random.seed!(42)

# PARAMETERS
const N  = 50
const T  = 10.0
const dt = 0.01
const nt = Int(round(T / dt))
const ntp1 = nt + 1
const ts = collect(range(0.0, T; length=ntp1))

global  wc = 0.0005
const we = 1.0
const t_prec_start = 2.0
const t_prec_end = T
const L_meas = t_prec_end - t_prec_start

@inline precision_window(t) = (t_prec_start <= t <= t_prec_end) ? 1.0 : 0.0
const w_prec_grid = [precision_window(t) for t in ts]


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

#Random.seed!(48882)

#edge_ind = sample(1:(N*N-N), 25, replace=false)


const edge_i = all_i[edge_ind]
const edge_j = all_j[edge_ind]
Adja = zeros(N,N)
for i in 1:length(edge_i)
    Adja[edge_i[i],edge_j[i]]=1.0
end

# NETWORK / CONTROLLED EDGES
#=
gra = watts_strogatz(N, 2, 0.001)
adj = Matrix(adjacency_matrix(gra))
edge_i = Int[]
edge_j = Int[]
for i in 1:N, j in 1:N
    if(i != j && adj[i, j] > 0)
        push!(edge_i, i)
        push!(edge_j, j)
    end
end
=#
const controlled_edges = collect(zip(edge_i, edge_j))

# Initial condition for x (one state per node)

#const Ji = 1.0*collect(-2.5:1.0:2.5)#randn(N)
const Ji = randn(N)

const x0 = randn(N) #Ji
# Linear interpolation of the time-dependent control matrix U.
@inline function time_index_and_weight(t)
    q = clamp(t / dt, 0.0, nt)
    idx = min(floor(Int, q) + 1, nt)
    return idx, q - (idx - 1)
end

function get_U_t!(Ut, t, U)
    idx, w = time_index_and_weight(t)
    @inbounds for j in 1:N, i in 1:N
        Ut[i, j] = (1 - w) * U[i, j, idx] + w * U[i, j, idx + 1]
        if(i==j)
            Ut[i,i] = 0.0
        end
    end
    return idx
end

mutable struct EvalWorkspace
    Ut_fwd::Matrix{Float64}
    Ut_adj::Matrix{Float64}
    X::Matrix{Float64}
    Lambda::Matrix{Float64}
    grad::Array{Float64,3}
end

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


EvalWorkspace() = EvalWorkspace(zeros(N,N), zeros(N,N),
                                zeros(N,ntp1), zeros(N,ntp1),
                                zeros(N,N,ntp1))

function evaluate_objective_and_gradient!(ws::EvalWorkspace, U)
    Ut_fwd, Ut_adj = ws.Ut_fwd, ws.Ut_adj
    X, Lambda, grad = ws.X, ws.Lambda, ws.grad
    fill!(grad, 0.0)
    function forward_ode!(dx, x, p, t)
        get_U_t!(Ut_fwd, t, U)
        @. dx = -x + p*tanh(x)
        mul!(dx, Ut_fwd, tanh.(x), 1.0, 1.0)
        return nothing
    end

    prob_x = ODEProblem(forward_ode!, x0, (0.0, T), Ji)
    sol_x = solve(prob_x, Tsit5(); reltol=1e-5, abstol=1e-7,
                  saveat=ts, save_everystep=false, dense=false)
    if !SciMLBase.successful_retcode(sol_x)
        error("Forward solve failed: $(sol_x.retcode)")
    end
    X .= Array(sol_x)


    function adjoint_ode!(dλ, λ, p, t)
        idx = get_U_t!(Ut_adj, t, U)
        xnow = @view X[:, idx]
        #q = transpose(Ut_adj) * λ
        #inp = sum(A[i] * cos(omega*t) + B[i] * sin(omega*t) for i=1:N)
        wp = precision_window(t)
        @inbounds for i in 1:N
                dλ[i] =  wp * we * (xnow[i, idx] - target_vals[i, idx])/N +  λ[i] - λ[i]*(p[i]*(1.0 - tanh(xnow[i])^2) + sum(Ut_adj[i,j]*(1.0 - tanh(xnow[j])^2) for j=1:N))
            end
        return nothing
    end

    λT = zeros(N)
    prob_λ = ODEProblem(adjoint_ode!, λT, (T, 0.0), Ji)
    sol_λ = solve(prob_λ, Tsit5(); reltol=1e-5, abstol=1e-7,
                  saveat=reverse(ts), save_everystep=false, dense=false)
    if !SciMLBase.successful_retcode(sol_λ)
        error("Adjoint solve failed: $(sol_λ.retcode)")
    end
    Lambda .= reverse(Array(sol_λ), dims=2)

    Fu = 0.0
    Fe = 0.0
    @inbounds for k in 1:ntp1
        wk = trapz_weight(k)
        Fu += wk * 0.5 * sum(abs2, @view U[:, :, k])

        s = 0.0
        for i in 1:N
            d = X[i, k] - target_vals[i, k]
            s += d*d
        end
        Fe += wk * 0.5 * w_prec_grid[k] * s /N
    end
    Fu *= dt
    Fe *= dt
    C = wc * Fu + we * Fe

    @inbounds for k in 1:ntp1, (i,j) in controlled_edges
        grad[i,j,k] = wc * U[i,j,k]/N^2 - Lambda[i,k] * tanh(X[j,k])
    end

    return C, wc*Fu, we*Fe, grad, X
end

# ADAM OPTIMIZER
const epochs = 30000
const η_adam = 0.001 #0.001
const β1 = 0.9
const β2 = 0.999
const ϵ = 1e-8

U = zeros(N, N, ntp1)
m = zeros(N, N, ntp1)
v = zeros(N, N, ntp1)
cost_hist = zeros(epochs)
Fu_hist = zeros(epochs)
Fe_hist = zeros(epochs)
grad_hist = zeros(epochs)
rate_hist = Vector{Matrix{Float64}}(undef, epochs)

ws = EvalWorkspace()

println("Starting Adam optimization for nonlinear neural units...")
#println("Target frequency: f_tar = $f_tar cycles per time unit")
@printf("%-7s | %-12s | %-12s | %-12s | %-12s\n",
        "Epoch", "Total Cost", "Fu (Ctrl)", "Fe ", "||Grad||")

for epoch in 1:epochs
   # if(epoch>4000)
   #     wc = 1.0
   # end
    C, Fu, Fe, grad, X = evaluate_objective_and_gradient!(ws, U)
    gnorm = norm(grad, Inf)
    cost_hist[epoch] = C
    Fu_hist[epoch] = Fu
    Fe_hist[epoch] = Fe
    grad_hist[epoch] = gnorm
    if(gnorm<0.0001 && epoch>100)
         @printf("%-6d | %-12.6f | %-12.6f | %-12.6f | %-12.6f\n",
                epoch, C, Fu, Fe, gnorm)
        ep = epoch 
        cost_hist[end] = C
        Fu_hist[end]   = Fu
        Fe_hist[end]   = Fe
        grad_hist[end] = gnorm
        rate_hist[end] = copy(X)
        break;
    end


    if epoch == 1 || epoch % 50 == 0
        @printf("%-7d | %-12.6f | %-12.6f | %-12.6f | %-12.6f\n",
                epoch, C, Fu, Fe, gnorm)
    end

    @. m = β1*m + (1-β1)*grad
    @. v = β2*v + (1-β2)*grad^2
    β1corr = 1 - β1^epoch
    β2corr = 1 - β2^epoch

    # Update only controlled edges; all other entries remain exactly zero.
    @inbounds for k in 1:ntp1, (i,j) in controlled_edges
        U[i,j,k] -= η_adam * (m[i,j,k] / β1corr) /
                    (sqrt(v[i,j,k] / β2corr) + ϵ)
    end
end

# Final evaluation
C_opt, Fu_opt, Fe_opt, grad_opt, X_opt =
    evaluate_objective_and_gradient!(ws, U)

println("------------------------------------------------------------")
println("Optimization complete.")
@printf("Final cost            : %.8f\n", C_opt)
@printf("Final control cost Fu : %.8f\n", Fu_opt)
@printf("Final Fourier cost Fe : %.8f\n", Fe_opt)
@printf("Final gradient norm   : %.8f\n", norm(grad_opt))

gf = 18
tf = 18
# Plot trajectories
pl_x = plot(ts, X_opt[1,:],  xlabel = "time [a.u.]",
    ylabel = L"x_i(t)",
    guidefontsize=18,
    label = L"x_1",
    linewidth = 2,
    tickfont=font(tf,"Computer Modern"), guidefont=font(gf,"Computer Modern")#,
   # title = "Optimal state trajectories"
)
#, xlabel="time [a.u.]", ylabel=L"x_i(t)",
 #           label="x₁", linewidth=2, title="Neural-unit states", legend=false)
for i in 2:N
    plot!(pl_x, ts, X_opt[i,:], label="x$i", linewidth=1.5, legend=false)
end


for i in 1:N
    plot!(pl_x, ts, target_vals[i, :], label = false, linestyle = :dash, linewidth = 1.5, color = "black", alpha=0.1)
end
vspan!(pl_x, [t_prec_start, t_prec_end], color = :gray, alpha = 0.15, label = "precision window")

title!(L"\mathcal{C}_e"*": $(round(Fe_opt,digits=5)); "*L"\mathcal{C}_u"*": $(round(Fu_opt,digits=4))",titlefont = font(24, "Computer Modern"))


U_flat = reshape(U, N*N, ntp1)
pl_weights = heatmap(ts, 1:(N*N), U_flat, xlabel="time [a.u.]",
                     ylabel="connection index", title="Controlled weights",
                     colorbar_title="Uᵢⱼ")

pl_selw = plot(xlabel = "time [a.u.]",tickfont=font(tf,"Computer Modern"), guidefont=font(gf,"Computer Modern"), ylabel = L"U_{ij}(t)")#, title = "Selected optimal weights")
for i in 1:min(N, 80), j in 1:min(N, 80)
    if(i!=j)
        plot!(pl_selw, ts, vec(U[i, j, :]), legend = false, label = "U[$i,$j]",guidefontsize=18, linewidth = 2)
    end
end

savefig(pl_x, "nonlinear_neural_states.png")
savefig(pl_weights, "nonlinear_neural_control_weights.png")
pl_all = plot(pl_x, pl_selw, layout=(2,1), size=(950, 800),tickfontsize=12)
savefig(pl_all, "nonlinear_neural_summary.png")
savefig(pl_all, "nonlinear_neural_summary.pdf")

display(pl_all)

pl_selw = plot(xlabel="time [a.u.]",ylabel=L"U_{ij}(t)")
for i in 1:min(N,80)
    for j in 1:min(N,80)
        if i != j
            plot!(pl_selw,ts, vec(U[i,j,:]),legend=false, label="U[$i,$j]", guidefontsize=18,linewidth=1.5)
        end
    end
end
