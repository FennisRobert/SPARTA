%EXAMPLE_SWEEP  Solve a complex-symmetric system with SPARTA, then sweep it.
%
%   The matrix is a 2D Laplacian with an imaginary shift on its diagonal, a
%   stand-in for a damped wave problem. Its pattern stays fixed while the
%   shift changes, so the sweep reuses the plan and the GPU memory and only
%   factors again.
%
%   Run build_sparta_mex once first.

side = 300;
L = delsq(numgrid('S', side + 2));          % side^2 unknowns, 5-point stencil
n = size(L, 1);
I = speye(n);
want = ones(n, 1);

fprintf('SPARTA %s on %s, %d unknowns\n', SpartaSolver.version(), SpartaSolver.device(), n);
if ~SpartaSolver.gpuAvailable()
    error('No GPU that SPARTA can use is visible to MATLAB.');
end

% One system: plan, factor and solve.
A = L + 0.05i * I;
b = A * want;
tic;
S = SpartaSolver(A, 'preset', 'sweep');
fprintf('plan + factor  %.3f s (GPU factor %.4f s), fill ratio %.1f, %s, %.1f MB of factors\n', ...
        toc, S.FactorSeconds, S.FillRatio, S.Kind, S.FactorBytes / 1e6);
x = S \ b;
fprintf('solve          relative residual %.1e, largest error %.1e\n', ...
        norm(b - A * x) / norm(b), max(abs(x - want)));

% A sweep: new values, same pattern, no new plan.
shifts = linspace(0.05, 1, 8);
tic;
for s = shifts
    A = L + 1i * s * I;
    factorize(S, A);
    x = S \ (A * want);
    fprintf('shift %.3f    factor %.4f s on the GPU, largest error %.1e\n', ...
            s, S.FactorSeconds, max(abs(x - want)));
end
fprintf('%d factor+solve pairs in %.3f s\n', numel(shifts), toc);

% Several right-hand sides at once are cheaper than one by one.
B = A * [want, 2 * want, 3 * want];
X = S \ B;
fprintf('3 right-hand sides, largest error %.1e\n', max(max(abs(X - [want, 2 * want, 3 * want]))));

% The single-precision answer without refinement, for comparison.
S.RefinementSteps = 0;
x = S \ b;
fprintf('unrefined      relative residual %.1e\n', norm(b - A * x) / norm(b));
