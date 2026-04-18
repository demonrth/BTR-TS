function [X, res, Out] = BTR_TS(Y, Omega, opts)

if isfield(opts, 'tol');         tol   = opts.tol;              else; tol = 1e-4;      end
if isfield(opts, 'maxit');       maxit = opts.maxit;            else; maxit = 150;     end
if isfield(opts, 'rho');         rho   = opts.rho;              else; rho = 1;         end
if isfield(opts, 'allR');        allR  = opts.allR;             else; error('opts.allR is required.'); end
if isfield(opts, 'beta1');       beta1 = opts.beta1;            else; beta1 = 1;       end
if isfield(opts, 'beta2');       beta2 = opts.beta2;            else; beta2 = 1;       end
if isfield(opts, 'alpha');       alpha = opts.alpha;            else; alpha = 1;       end
if isfield(opts, 'lambda_t');    lambda_t = opts.lambda_t;      else; lambda_t = 0;    end
if isfield(opts, 'x_inner_tol'); x_inner_tol = opts.x_inner_tol; else; x_inner_tol = 1e-6; end
if isfield(opts, 'x_inner_maxit'); x_inner_maxit = opts.x_inner_maxit; else; x_inner_maxit = 100; end

Nway = size(Y);

% --- Precompute temporal Laplacian on the merged intra-day time axis ---
Lt = [];
if lambda_t > 0
    T = Nway(3) * Nway(4);
    if T > 1
        Lt = local_time_laplacian_sparse(T);
    end
end

obs_mask = false(Nway);
obs_mask(Omega) = true;

X = Y;

R1=allR(1); R2=allR(2); R=allR(3);

G = cell(1,3);
G{1} = ones(R1,Nway(1),R1);
G{2} = ones(R1,Nway(2),R1);
G{3} = ones(R1,R,R1);

F = cell(1,3);
F{1} = ones(R2,R,R2);
F{2} = ones(R2,Nway(3),R2);
F{3} = ones(R2,Nway(4),R2);

A = ones(Nway(1),Nway(2),R);
B = ones(R,Nway(3),Nway(4));

Out.Res = [];
Out.X_inner_it = [];

for k = 1:maxit
    Xold = X;

    %% Update A
    Xi = reshape(X,[prod([Nway(1),Nway(2)]),prod([Nway(3),Nway(4)])]);
    Ai = reshape(A,[],R);
    Bi = reshape(B,R,[]);
    C  = TR_ybz(G);
    Ci = reshape(C,[],R);
    tempC = alpha*Xi*Bi'+beta1*Ci+rho*Ai;
    tempA = alpha*(Bi*Bi')+(rho+beta1)*eye(R);
    A     = reshape(tempC*pinv(tempA),size(A));

    %% Update B
    Ai = reshape(A,[],R);
    D  = TR_ybz(F);
    Di = reshape(D,R,[]);
    tempC = alpha*Ai'*Xi+beta2*Di+rho*Bi;
    tempA = alpha*(Ai'*Ai)+(rho+beta2)*eye(R);
    B = reshape(pinv(tempA)*tempC,size(B));

    %% Update G_1,2,3
    for i = 1:3
        Gi = Unfold(G{i},size(G{i}),2);
        Ai = Unfold(A,size(A),i);
        Mi = BTR_rest_ybz(G, i);
        tempC = beta1*Ai*Mi'+rho*Gi;
        tempA = beta1*(Mi*Mi')+rho*eye(size(Gi,2));
        G{i}  = Fold(tempC*pinv(tempA),size(G{i}),2);
    end

    %% Update F_1,2,3
    for i = 1:3
        Fi = Unfold(F{i},size(F{i}),2);
        Bi = Unfold(B,size(B),i);
        Mi = BTR_rest_ybz(F, i);
        tempC = beta2*Bi*Mi'+rho*Fi;  
        tempA = beta2*(Mi*Mi')+rho*eye(size(Fi,2));
        F{i}  = Fold(tempC*pinv(tempA),size(F{i}),2);
    end

    %% Update X 
    AB = tensor_contraction_ybz(A,B,3,1);
    V  = (alpha*AB + rho*Xold) / (alpha + rho);

    if lambda_t > 0 && ~isempty(Lt)
        [X, inner_it] = solve_X_subproblem_unified(V, Xold, Y, obs_mask, Lt, alpha + rho, lambda_t, x_inner_maxit, x_inner_tol);
    else
        X = V;
        X(Omega)= Y(Omega);
        inner_it = 0;
    end

    %% check the convergence
    res = norm(X(:)-Xold(:)) / max(norm(Xold(:)), eps);
    Out.Res = [Out.Res, res];
    Out.X_inner_it = [Out.X_inner_it, inner_it];

    if res < tol
        break;
    end
end
end

%% ====== Nested Sub-functions ======
function L = local_time_laplacian_sparse(n)
% L = D' * D for first-order 1D differences with natural end points.
    if n <= 1
        L = sparse(1,1,0,1,1);
        return;
    end
    main = 2 * ones(n,1);
    main(1) = 1;
    main(n) = 1;
    off  = -ones(n-1,1);
    L = spdiags([[off;0], main, [0;off]], [-1 0 1], n, n);
end

function [X, it_used] = solve_X_subproblem_unified(V, Xwarm, Y, obs_mask, Lt, mu, lambda_t, maxit_cg, tol_cg)
    Nway = size(V);
    T = Nway(3) * Nway(4);
    M = Nway(1) * Nway(2);

    Vmat = reshape(permute(V,     [3 4 1 2]), T, M);
    Ymat = reshape(permute(Y,     [3 4 1 2]), T, M);
    Omat = reshape(permute(obs_mask, [3 4 1 2]), T, M);
    Xw   = reshape(permute(Xwarm, [3 4 1 2]), T, M);

    Xfix = zeros(T, M);
    Xfix(Omat) = Ymat(Omat);

    rhs = mu * (Vmat - Xfix) - lambda_t * (Lt * Xfix);
    rhs(Omat) = 0; 

    W = Xw - Xfix;
    W(Omat) = 0;

    Aop = @(Z) apply_missing_hessian(Z, Lt, Omat, mu, lambda_t);

    R = rhs - Aop(W);
    P = R;
    rhs_norm = norm(rhs, 'fro');

    if rhs_norm <= max(tol_cg, eps)
        it_used = 0;
    else
        rsold = sum(R(:) .* R(:));
        it_used = 0;
        for it = 1:maxit_cg
            AP = Aop(P);
            denom = sum(P(:) .* AP(:));
            if abs(denom) < eps
                break;
            end
            alpha_cg = rsold / denom;
            W = W + alpha_cg * P;
            W(Omat) = 0;

            R = R - alpha_cg * AP;
            rsnew = sum(R(:) .* R(:));
            it_used = it;

            if sqrt(rsnew) / rhs_norm < tol_cg
                break;
            end

            beta_cg = rsnew / max(rsold, eps);
            P = R + beta_cg * P;
            P(Omat) = 0;
            rsold = rsnew;
        end
    end

    Xmat = Xfix + W;
    Xmat(Omat) = Ymat(Omat);
    X = permute(reshape(Xmat, [Nway(3), Nway(4), Nway(1), Nway(2)]), [3 4 1 2]);
end

function AW = apply_missing_hessian(W, Lt, Omat, mu, lambda_t)
    W(Omat) = 0;
    AW = mu * W + lambda_t * (Lt * W);
    AW(Omat) = 0;
end

function Out = TR_ybz(G)
    Out = tensor_contraction_ybz(tensor_contraction_ybz(G{1},G{2},3,1),G{3},[4,1],[1,3]);
end

function Out = BTR_rest_ybz(G,i)
    n=[1,4]; m=[2,3];
    if i == 1
        GI=tensor_contraction_ybz(G{2},G{3},3,1);
    end
    if i == 2
        GI=tensor_contraction_ybz(G{3},G{1},3,1);
    end
    if i == 3
        GI=tensor_contraction_ybz(G{1},G{2},3,1);
    end
    NwayOut = size(GI);
    GI= permute(GI,[n,m]);
    Out = reshape(GI,prod(NwayOut(n)),prod(NwayOut(m)));
end

function W = Unfold(W, dim, i)
    W = reshape(shiftdim(W,i-1), dim(i), []);
end

function W = Fold(W, dim, i)
    dim = circshift(dim, [1-i, 1-i]);
    W = shiftdim(reshape(W, dim), length(dim)+1-i);
end