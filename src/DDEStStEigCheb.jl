"""
Eigenvalues of constant-coefficient linear DDE using Chebyshev interpolation.

Translated from DDE-Biftool (MATLAB) `dde_stst_eig_cheb.m`.
"""
module DDEStStEigCheb

using LinearAlgebra


export eigenvalues_chebyshev, ChebyshevOptions, StabilityResult, DiscardedRoots

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

"""
Eigenvalues discarded due to large residual.
"""
struct DiscardedRoots
    eigenvalues::Vector{ComplexF64}   # na literatura geralmente denotado por lambda
    errors::Vector{Float64}
end

"""
Result of eigenvalue computation for a steady-state of a DDE.
"""
mutable struct StabilityResult
    eigenvalues::Vector{ComplexF64}       # na literatura: lambda_0
    corrected_eigenvalues::Vector{ComplexF64}  # na literatura: lambda_1
    errors::Vector{Float64}
    right_eigenvectors::Array{ComplexF64}  # na literatura: v
    left_eigenvectors::Array{ComplexF64}   # na literatura: w
    mesh::Union{Nothing,Vector{Float64}}
    discarded::Union{Nothing,DiscardedRoots}
end

function StabilityResult(;
    eigenvalues=ComplexF64[],
    corrected_eigenvalues=ComplexF64[],
    errors=Float64[],
    right_eigenvectors=Matrix{ComplexF64}(undef, 0, 0),
    left_eigenvectors=Matrix{ComplexF64}(undef, 0, 0),
    mesh=nothing,
    discarded=nothing,
)
    StabilityResult(
        eigenvalues, corrected_eigenvalues, errors,
        right_eigenvectors, left_eigenvectors, mesh, discarded,
    )
end

"""
Options for `eigenvalues_chebyshev`.
"""
Base.@kwdef struct ChebyshevOptions
    ncheb::Union{Nothing,Int} = nothing
    minimal_real_part::Float64 = -1.0
    root_accuracy::Float64 = 1e-6
    max_matrix_size::Int = 200
    initial_matrix_size::Union{Nothing,Int} = nothing   # calculado a partir de dim se nothing
    ncheb_growth_factor::Float64 = 2.0
    max_number_of_eigenvalues::Int = 20
    min_number_of_eigenvalues::Union{Nothing,Int} = nothing
    scale_left_eigenvectors::Bool = true       # na literatura: escala w
    closest::Union{Nothing,ComplexF64} = nothing
    delay_scale_limit::Float64 = 1e-5       # tauscal_limit
    imag_threshold::Float64 = 1e-10
    discard_accuracy_factor::Float64 = 1e5
    lhs_matrix::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)  # preenchido na chamada
    return_full_eigenvectors::Bool = false       # VWout
end

# ---------------------------------------------------------------------------
# Funcao publica principal
# ---------------------------------------------------------------------------

"""
    eigenvalues_chebyshev(coeff_matrices, delays; kw...) -> StabilityResult

Compute eigenvalues of the constant-coefficient linear DDE

    lhs_matrix * x'(t) = sum_k  A_k * x(t - tau_k)

using Chebyshev discretization on the interval [-tau_max, 0].

# Arguments
- `coeff_matrices::Array{Float64,3}`: system matrices ``A_k`` stacked along 3rd dimension (dim x dim x n_delays).
- `delays::Vector{Float64}`: delay values ``tau_k``.

Keyword arguments map to [`ChebyshevOptions`](@ref).
"""
function eigenvalues_chebyshev(
    coeff_matrices::Array{Float64,3},  # na literatura: A_k
    delays::Vector{Float64};           # na literatura: tau
    kwargs...,
)
    dim, _, num_matrices = size(coeff_matrices)

    # Ajustar vetor de atrasos: aceita com ou sem tau_0 = 0
    if length(delays) == num_matrices - 1
        delays = [0.0; delays]
    elseif length(delays) != num_matrices
        error(
            "Numero de matrizes ($num_matrices) incompativel com " *
            "numero de atrasos ($(length(delays))).",
        )
    end

    delays = max.(0.0, delays)

    # Montar opcoes com defaults dependentes de dim
    opts = build_options(dim; kwargs...)

    max_delay = maximum(delays)
    result = StabilityResult()
    if opts.return_full_eigenvectors
        result.mesh = Float64[]
    end

    # Estimativa de escala do problema
    coefficient_norm_estimate = sum(
        opnorm(coeff_matrices[:, :, k], Inf) * abs(delays[k]) for k in 1:num_matrices
    )

    # --- Caso especial: todos os atrasos sao zero ---
    if coefficient_norm_estimate == 0.0
        return solve_zero_delay_case!(
            result, coeff_matrices, delays, opts, dim,
        )
    end

    # --- Atrasos muito pequenos: escalar ate delay_scale_limit ---
    if max_delay < opts.delay_scale_limit
        delays = [delays; opts.delay_scale_limit]
        coeff_matrices = cat(coeff_matrices, zeros(dim, dim, 1), dims=3)
        max_delay = opts.delay_scale_limit
    end

    scale = 2.0 / max_delay
    scaled_delays = delays .* scale
    scaled_matrices = coeff_matrices ./ scale   # na literatura: A / sigma

    # Escolha inicial de pontos de Chebyshev
    ncheb = choose_initial_ncheb(coeff_matrices, delays, opts, dim)

    # Duplicar ncheb ate erro aceitavel ou limite
    eigenvals = ComplexF64[]
    right_vecs = Matrix{ComplexF64}(undef, dim, 0)
    left_vecs = Matrix{ComplexF64}(undef, dim, 0)
    full_right = Matrix{ComplexF64}(undef, 0, 0)
    full_left = Matrix{ComplexF64}(undef, 0, 0)
    residuals = Matrix{ComplexF64}(undef, dim, 0)
    nodes = Float64[]
    err = Inf

    while err > opts.root_accuracy && ncheb <= opts.max_matrix_size / dim
        diff_matrix, interp_at_delays, nodes =
            chebyshev_diff_and_interp(ncheb, scaled_delays)

        # Montar o operador discretizado
        operator = kron(diff_matrix, Matrix{Float64}(I, dim, dim))
        # Substituir primeiras dim linhas pela condicao de contorno
        operator[1:dim, :] .= 0.0
        for k in 1:num_matrices
            operator[1:dim, :] .+=
                scaled_matrices[:, :, k] * kron(interp_at_delays[k:k, :], Matrix{Float64}(I, dim, dim))
        end

        eigenvals, right_vecs, left_vecs, full_right, full_left =
            sorted_eigenvalues(operator, opts, dim, scale)

        residuals, left_vecs, full_left = compute_residuals(
            eigenvals, right_vecs, left_vecs, full_left,
            coeff_matrices, delays, opts,
        )

        finite_res = filter(isfinite, residuals)
        err = isempty(finite_res) ? 0.0 : maximum(abs, finite_res)
        ncheb = round(Int, ncheb * opts.ncheb_growth_factor)
    end

    result.eigenvalues = eigenvals
    result.corrected_eigenvalues = copy(eigenvals)
    result.errors = vec(maximum(abs, residuals; dims=1))
    result.right_eigenvectors = right_vecs
    result.left_eigenvectors = left_vecs

    if opts.return_full_eigenvectors
        num_eig = length(eigenvals)
        result.right_eigenvectors = reshape(full_right, dim, :, num_eig)
        result.left_eigenvectors = reshape(full_left, dim, :, num_eig)
        result.mesh = nodes ./ scale
    end

    discard_inaccurate_roots!(result, opts)
    return result
end


function eigenvalues_chebyshev(
    delay_to_matrix::Dict{Float64,Matrix{Float64}};
    kwargs...,
)
    sorted_delays = sort(collect(keys(delay_to_matrix)))

    coeff_matrices = cat(
        (delay_to_matrix[d] for d in sorted_delays)...;
        dims=3,
    )
    return eigenvalues_chebyshev(coeff_matrices, sorted_delays; kwargs...)
end

# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------

"""Preenche opcoes com defaults que dependem de `dim`."""
function build_options(dim::Int; kwargs...)
    kw = Dict{Symbol,Any}(kwargs)

    initial_size = get(kw, :initial_matrix_size, nothing)
    if initial_size === nothing
        initial_size = max(2, ceil(Int, 40 / dim))
    end
    kw[:initial_matrix_size] = initial_size

    min_eig = get(kw, :min_number_of_eigenvalues, nothing)
    if min_eig === nothing
        kw[:min_number_of_eigenvalues] = dim
    end

    lhs = get(kw, :lhs_matrix, nothing)
    if lhs === nothing || size(lhs) == (0, 0)
        kw[:lhs_matrix] = Matrix{Float64}(I, dim, dim)
    end

    return ChebyshevOptions(;
        (k => v for (k, v) in kw if hasfield(ChebyshevOptions, k))...,
    )
end

"""Resolve caso especial onde todos os atrasos sao efetivamente zero."""
function solve_zero_delay_case!(
    result::StabilityResult,
    coeff_matrices::Array{Float64,3},
    delays::Vector{Float64},
    opts::ChebyshevOptions,
    dim::Int,
)
    zero_idx = findall(d -> d == 0.0, delays)
    sum_matrix = sum(coeff_matrices[:, :, k] for k in zero_idx)

    eigenvals, right_vecs, left_vecs, _, full_left =
        sorted_eigenvalues(sum_matrix, opts, dim, 1.0)

    result.eigenvalues = eigenvals
    result.corrected_eigenvalues = copy(eigenvals)

    residuals, left_vecs, _ = compute_residuals(
        eigenvals, right_vecs, left_vecs, full_left,
        coeff_matrices, delays, opts,
    )
    result.errors = vec(maximum(abs, residuals; dims=1))
    result.right_eigenvectors = right_vecs
    result.left_eigenvectors = left_vecs

    if opts.return_full_eigenvectors
        result.mesh = [0.0]
    end

    discard_inaccurate_roots!(result, opts)
    return result
end

"""Escolhe numero inicial de pontos de Chebyshev."""
function choose_initial_ncheb(
    coeff_matrices::Array{Float64,3},
    delays::Vector{Float64},
    opts::ChebyshevOptions,
    dim::Int,
)::Int
    if opts.ncheb !== nothing
        ncheb = opts.ncheb
    else
        ncheb = max(4.0, opnorm(coeff_matrices[:, :, 1]))
        if isfinite(opts.minimal_real_part)
            for k in 2:length(delays)
                ncheb += opnorm(coeff_matrices[:, :, k]) *
                         exp(abs(opts.minimal_real_part) * delays[k])
            end
        end
        ncheb = round(Int, ncheb)
    end

    min_eig = something(opts.min_number_of_eigenvalues, dim)
    ini_size = something(opts.initial_matrix_size, max(2, ceil(Int, 40 / dim)))

    ncheb = max(
        ceil(Int, min(ncheb, ini_size / dim)),
        ceil(Int, min_eig / dim),
    )
    return ncheb
end

# ---------------------------------------------------------------------------
# Chebyshev: nos, pesos baricentricos, diferenciacao e interpolacao
# ---------------------------------------------------------------------------

"""
    chebyshev_nodes_and_weights(degree, interval) -> (nodes, bary_weights)

Chebyshev nodes and barycentric weights on `interval`.
"""
function chebyshev_nodes_and_weights(degree::Int, interval::Tuple{Float64,Float64}=(-1.0, 1.0))
    len = interval[2] - interval[1]
    js = 0:degree
    # Nos de Chebyshev em [-1, 1]
    nodes_unit = cos.(js .* (pi / degree))
    # Pesos baricentricos
    bary_weights = ones(degree + 1)
    for j in js
        if isodd(j)
            bary_weights[j+1] = -1.0
        end
    end
    bary_weights[1] *= 0.5
    bary_weights[end] *= 0.5

    # Escalar para intervalo desejado
    nodes = @. (nodes_unit + 1) / 2 * len + interval[1]
    bary_weights .*= (2.0 / len)^degree

    # Inverter para ordem crescente
    reverse!(nodes)
    reverse!(bary_weights)

    return nodes, bary_weights
end

"""
    chebyshev_diff_and_interp(ncheb, scaled_delays)
        -> (diff_matrix, interp_at_delays, nodes)

Construct Chebyshev differentiation matrix and barycentric interpolation
matrices at the given delay points.

Returns:
- `diff_matrix`: (ncheb+1) x (ncheb+1) differentiation matrix  # na literatura: D
- `interp_at_delays`: (n_delays) x (ncheb+1) interpolation matrix at -tau  # na literatura: E(tau)
- `nodes`: Chebyshev nodes in physical scale
"""
function chebyshev_diff_and_interp(ncheb::Int, scaled_delays::Vector{Float64})
    num_points = ncheb + 1
    tau_max = maximum(abs, scaled_delays)

    # chebyshev_nodes_and_weights retorna em ordem crescente [-1..1].
    # Reverter para ordem decrescente [1..-1] para que nodes[1] = 0 (presente)
    # e nodes[end] = -tau_max (passado), compativel com a condicao de contorno.
    nodes, bary_weights = chebyshev_nodes_and_weights(ncheb, (-1.0, 1.0))
    reverse!(nodes)
    reverse!(bary_weights)
    bary_weights .*= (2.0 / tau_max)^ncheb
    nodes = @. (nodes - 1.0) / 2.0 * tau_max

    num_delays = length(scaled_delays)

    # --- Interpolacao baricentrica em -tau ---
    interp_at_delays = zeros(num_delays, num_points)
    for d in 1:num_delays
        eval_point = -scaled_delays[d]
        weights_over_diff = similar(bary_weights)
        exact_node = -1  # indice se eval_point coincide com um no
        for j in 1:num_points
            diff = eval_point - nodes[j]
            if diff == 0.0
                exact_node = j
                break
            end
            weights_over_diff[j] = bary_weights[j] / diff
        end
        if exact_node > 0
            interp_at_delays[d, :] .= 0.0
            interp_at_delays[d, exact_node] = 1.0
        else
            denom = sum(weights_over_diff)
            interp_at_delays[d, :] .= weights_over_diff ./ denom
        end
    end

    # --- Matriz de diferenciacao de Chebyshev (baricentrica) ---
    diff_matrix = zeros(num_points, num_points)
    for i in 1:num_points
        for j in 1:num_points
            if i != j
                diff_matrix[i, j] = bary_weights[j] / bary_weights[i] /
                                    (nodes[i] - nodes[j])
            end
        end
        diff_matrix[i, i] = -sum(diff_matrix[i, k] for k in 1:num_points if k != i)
    end

    return diff_matrix, interp_at_delays, nodes
end

# ---------------------------------------------------------------------------
# Autovalores: calculo, ordenacao e selecao
# ---------------------------------------------------------------------------

"""
    sorted_eigenvalues(operator, opts, dim, scale)
        -> (eigenvalues, right_vecs, left_vecs, full_right, full_left)

Compute eigenvalues of `operator` (possibly generalized), sort by real part
(descending) or proximity to `opts.closest`, and select up to
`max_number_of_eigenvalues`.
"""
function sorted_eigenvalues(
    operator::AbstractMatrix{Float64},
    opts::ChebyshevOptions,
    dim::Int,
    scale::Float64,
)
    lhs = generalized_lhs_matrix(operator, opts.lhs_matrix)

    # Usar LAPACK diretamente para obter autovetores esquerdos e direitos
    # ja emparelhados em uma unica chamada (equivalente ao [V,D,W]=eig(A) do MATLAB).
    # Convertemos para ComplexF64 para evitar o formato compactado real do dgeev/dggev.
    if lhs === nothing
        W, VL, VR = LAPACK.geev!('V', 'V', ComplexF64.(operator))
        raw_eigenvalues = ComplexF64.(W .* scale)
        full_right = VR
        full_left = VL
    else
        alpha, beta, VL, VR = LAPACK.ggev!('V', 'V', ComplexF64.(operator), ComplexF64.(lhs))
        raw_eigenvalues = ComplexF64.(alpha ./ beta .* scale)
        full_right = VR
        full_left = VL
    end

    # Substituir nao-finitos por -Inf
    replace!(x -> isfinite(x) ? x : complex(-Inf), raw_eigenvalues)

    # Ordenar por parte imaginaria descendente (estabilidade de ordenacao)
    perm_imag = sortperm(imag.(raw_eigenvalues); rev=true)
    raw_eigenvalues = raw_eigenvalues[perm_imag]
    full_right = full_right[:, perm_imag]
    full_left = full_left[:, perm_imag]

    # Ordenar por criterio principal
    if opts.closest === nothing
        perm = sortperm(real.(raw_eigenvalues); rev=true)
        selected = real.(raw_eigenvalues[perm]) .>= opts.minimal_real_part
    else
        perm = sortperm(abs.(raw_eigenvalues .- opts.closest))
        selected = trues(length(raw_eigenvalues))
    end

    raw_eigenvalues = raw_eigenvalues[perm]
    full_right = full_right[:, perm]
    full_left = full_left[:, perm]

    min_eig = something(opts.min_number_of_eigenvalues, dim)

    # Garantir minimo de autovalores selecionados
    for k in 1:min(length(raw_eigenvalues), min_eig)
        selected[k] = true
    end

    # Nao descartar conjugado complexo do ultimo selecionado
    if length(selected) > 1 && !all(selected)
        last_sel = findlast(selected)
        if last_sel !== nothing && last_sel < length(selected)
            if abs(real(raw_eigenvalues[last_sel]) - real(raw_eigenvalues[last_sel+1])) < opts.imag_threshold
                selected[last_sel+1] = true
            end
        end
    end

    max_eig = min(opts.max_number_of_eigenvalues, length(selected))
    selected[max_eig+1:end] .= false

    eigenvalues = raw_eigenvalues[selected]
    right_sel = full_right[:, selected]
    left_sel = full_left[:, selected]

    # Extrair primeiras dim componentes como autovetores fisicos
    right_vecs = right_sel[1:dim, :]
    left_vecs = left_sel[1:dim, :]

    # Normalizar autovetores direitos
    norms = sqrt.(vec(sum(right_vecs .* conj.(right_vecs); dims=1)))
    norms[norms.==0] .= 1.0
    inv_norms = Diagonal(1.0 ./ norms)
    right_vecs = right_vecs * inv_norms
    right_sel = right_sel * inv_norms

    # Preencher se menos que o minimo
    if length(eigenvalues) < min_eig
        deficit = min_eig - length(eigenvalues)
        append!(eigenvalues, fill(complex(-Inf), deficit))
        right_vecs = hcat(right_vecs, fill(complex(NaN), dim, deficit))
        left_vecs = hcat(left_vecs, fill(complex(NaN), dim, deficit))
    end

    return eigenvalues, right_vecs, left_vecs, right_sel, left_sel
end



"""Constroi matriz LHS para problema generalizado, ou retorna nothing se for identidade."""
function generalized_lhs_matrix(
    operator::AbstractMatrix,
    lhs_matrix::AbstractMatrix{Float64},
)
    if norm(lhs_matrix - I, Inf) == 0.0
        return nothing
    end
    op_size = size(operator, 1)
    lhs_size = size(lhs_matrix, 1)
    return cat(lhs_matrix, Matrix{Float64}(I, op_size - lhs_size, op_size - lhs_size); dims=(1, 2))
end

# ---------------------------------------------------------------------------
# Matriz caracteristica e residuos
# ---------------------------------------------------------------------------

"""
    characteristic_matrix(coeff_matrices, delays, lambda; lhs_matrix, derivative_order)

Compute the characteristic matrix Delta(lambda) of the DDE.

    Delta(lambda) = lambda^(deri) * lhs_matrix - sum_k (-tau_k)^deri * A_k * exp(-lambda * tau_k)

Na literatura geralmente denotado por Delta(lambda).
"""
function characteristic_matrix(
    coeff_matrices::Array{Float64,3},  # na literatura: A_k
    delays::Vector{Float64},           # na literatura: tau
    lambda::Number;
    lhs_matrix::AbstractMatrix{Float64}=Matrix{Float64}(I, size(coeff_matrices, 1), size(coeff_matrices, 1)),
    derivative_order::Int=0,         # na literatura: deri
)
    dim = size(coeff_matrices, 1)
    num_matrices = size(coeff_matrices, 3)

    taus = length(delays) < num_matrices ? [0.0; delays] : copy(delays)

    # Na literatura: lfac = [lambda, 1, 0, ...] indexado por deri+1
    # deri=0 -> lambda,  deri=1 -> 1,  deri>=2 -> 0
    lambda_factor = if derivative_order == 0
        lambda
    elseif derivative_order == 1
        one(lambda)
    else
        zero(lambda)
    end
    tau_sign = (-1)^derivative_order

    delta = complex(lambda_factor) .* complex.(lhs_matrix)
    for k in 1:num_matrices
        delta .-= tau_sign * taus[k]^derivative_order *
                  coeff_matrices[:, :, k] .* exp(-lambda * taus[k])
    end
    return delta
end

"""
    compute_residuals(eigenvalues, right_vecs, left_vecs, full_left,
                      coeff_matrices, delays, opts)
        -> (residuals, left_vecs, full_left)

Compute residual Delta(lambda)*v for each eigenvalue, and optionally scale
left eigenvectors.
"""
function compute_residuals(
    eigenvalues::Vector{ComplexF64},
    right_vecs::Matrix{ComplexF64},     # na literatura: v
    left_vecs::Matrix{ComplexF64},      # na literatura: w
    full_left::Matrix{ComplexF64},      # na literatura: W (full)
    coeff_matrices::Array{Float64,3},
    delays::Vector{Float64},
    opts::ChebyshevOptions,
)
    dim = size(coeff_matrices, 1)
    num_eig = length(eigenvalues)
    residuals = fill(complex(Inf), dim, num_eig)

    # Copia mutavel para escalar
    left_vecs = copy(left_vecs)
    full_left = copy(full_left)

    for i in num_eig:-1:1
        isfinite(eigenvalues[i]) || continue
        delta = characteristic_matrix(
            coeff_matrices, delays, eigenvalues[i];
            lhs_matrix=opts.lhs_matrix,
        )
        delta_deriv_v = characteristic_matrix(
            coeff_matrices, delays, eigenvalues[i];
            lhs_matrix=opts.lhs_matrix, derivative_order=1,
        ) * right_vecs[:, i]

        if opts.scale_left_eigenvectors
            scale_factor = dot(left_vecs[:, i], delta_deriv_v)
            if scale_factor != 0.0
                left_vecs[:, i] ./= scale_factor
                full_left[:, i] ./= scale_factor
            end
        end

        residuals[:, i] = delta * right_vecs[:, i]
    end

    return residuals, left_vecs, full_left
end

# ---------------------------------------------------------------------------
# Pos-processamento: descartar raizes imprecisas
# ---------------------------------------------------------------------------

"""Remove eigenvalues whose residual exceeds the tolerance threshold."""
function discard_inaccurate_roots!(result::StabilityResult, opts::ChebyshevOptions)
    threshold = opts.root_accuracy * opts.discard_accuracy_factor
    keep = result.errors .<= threshold

    if any(.!keep)
        result.discarded = DiscardedRoots(
            result.eigenvalues[.!keep],
            result.errors[.!keep],
        )
    end

    result.eigenvalues = result.eigenvalues[keep]
    result.corrected_eigenvalues = result.corrected_eigenvalues[keep]
    result.errors = result.errors[keep]

    if ndims(result.right_eigenvectors) == 3
        result.right_eigenvectors = result.right_eigenvectors[:, :, keep]
        result.left_eigenvectors = result.left_eigenvectors[:, :, keep]
    else
        result.right_eigenvectors = result.right_eigenvectors[:, keep]
        result.left_eigenvectors = result.left_eigenvectors[:, keep]
    end

    return result
end

end # module
