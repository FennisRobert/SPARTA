function test_sparta()
%TEST_SPARTA  Check that SpartaSolver works in this MATLAB.
%
%   Run build_sparta_mex first. Prints one line per check and stops at the
%   first failure. Please report a failure, with its message, as an issue.

assert(SpartaSolver.gpuAvailable(), 'no GPU that SPARTA can use is visible to MATLAB');
fprintf('SPARTA %s on %s\n', SpartaSolver.version(), SpartaSolver.device());

L = delsq(numgrid('S', 62));
n = size(L, 1);
I = speye(n);
want = (1:n).' / n + 0.5i;
residual = @(A, x, b) norm(b - A * x) / norm(b);

% Complex-symmetric, detected as such.
A = L + 0.05i * I;
S = SpartaSolver(A);
check(strcmp(S.Kind, 'symmetric'), 'symmetric matrix detected');
check(residual(A, S \ (A * want), A * want) < 1e-6, 'symmetric solve, refined');
S.RefinementSteps = 0;
r = residual(A, S \ (A * want), A * want);
check(r < 1e-4 && r > 1e-9, sprintf('single-precision solve (%.1e)', r));
S.RefinementSteps = -1;

% Several right-hand sides, real and complex.
B = [A * want, real(A * want), zeros(n, 1)];
X = S \ B;
check(isequal(size(X), [n, 3]) && residual(A, X(:, 1), B(:, 1)) < 1e-6 ...
      && all(X(:, 3) == 0), 'three right-hand sides, one of them zero');

% A sweep: new values on the same pattern.
for s = [0.2, 0.7]
    A = L + 1i * s * I;
    factorize(S, A);
    check(residual(A, S \ (A * want), A * want) < 1e-6, sprintf('refactor at shift %.1f', s));
end

% Entries that come out exactly zero are dropped by MATLAB; the plan stays.
id = S.FillRatio;
Az = A;
Az(2, 1) = 0;
Az(1, 2) = 0;
factorize(S, Az);
check(residual(Az, S \ (Az * want), Az * want) < 1e-6 && S.FillRatio == id, 'entries dropped, same plan');

% A pattern that grew is planned again.
Ag = A;
Ag(n, 1) = -0.01;
Ag(1, n) = -0.01;
factorize(S, Ag);
check(residual(Ag, S \ (Ag * want), Ag * want) < 1e-6, 'grown pattern planned again');

% A general matrix.
U = L + 0.05i * I + sparse(2:n, 1:n - 1, 0.3, n, n);
G = SpartaSolver(U, 'preset', 'careful', 'nemin', 24);
check(strcmp(G.Kind, 'general'), 'general matrix detected');
check(residual(U, G \ (U * want), U * want) < 1e-6, 'general solve');

% Symmetric values turning unsymmetric in a sweep switch to the general path.
factorize(S, U);
check(strcmp(S.Kind, 'general') && residual(U, S \ (U * want), U * want) < 1e-6, ...
      'automatic kind follows the matrix');

% A real matrix and a real right-hand side.
R = SpartaSolver(L);
check(residual(L, R \ ones(n, 1), ones(n, 1)) < 1e-6, 'real matrix');

% Errors carry an identifier.
check(fails(@() SpartaSolver(A, 'nemn', 3), 'sparta:invalid'), 'unknown setting refused');
check(fails(@() SpartaSolver(A, 'preset', 'fast'), 'sparta:invalid'), 'unknown preset refused');
check(fails(@() S \ ones(n + 1, 1), 'sparta:invalid'), 'wrong right-hand side size refused');
check(fails(@() SpartaSolver(sparse(n, n)), 'sparta:pivot'), 'singular matrix reported');

% Releasing.
clear S G R
check(true, 'solvers released');
fprintf('All checks passed.\n');
end

function check(ok, what)
% Prints one passed check, or stops at a failed one.
if ~ok
    error('sparta:test', 'FAILED: %s', what);
end
fprintf('  ok  %s\n', what);
end

function tf = fails(f, id)
% True when calling f raises an error with identifier id.
tf = false;
try
    f();
catch err
    tf = strcmp(err.identifier, id);
end
end
