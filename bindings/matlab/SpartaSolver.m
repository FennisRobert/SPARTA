classdef SpartaSolver < handle
%SPARTASOLVER  Solve sparse linear systems on the GPU with SPARTA.
%
%   S = SpartaSolver(A) plans and factors the square sparse matrix A.
%   x = S \ b solves A x = b. b may hold several right-hand sides as columns.
%
%   factorize(S, A2) factors a new matrix with the same pattern of nonzeros,
%   reusing the plan and the GPU memory. This is the frequency-sweep case: the
%   planning is paid for once. A matrix whose pattern has grown is planned
%   again automatically.
%
%   The GPU computes in single precision. By default the answer is refined in
%   double precision until the relative residual norm(b - A*x)/norm(b) reaches
%   1e-6, which on a well-conditioned system usually means one extra solve.
%
%   S = SpartaSolver(A, name, value, ...) changes a setting:
%
%     'kind'              'auto' (default) tests whether A equals its
%                         transpose and picks 'symmetric' or 'general'.
%                         'symmetric' factors A = L D L.' and reads only the
%                         lower triangle; 'general' factors A = L U.
%     'preset'            Starting point for the settings below: 'balanced'
%                         (default), 'sweep', 'one_shot', 'structured_grid' or
%                         'careful'.
%     'refinement_steps'  Most corrections per solve. -1 (default) corrects
%                         until 'refinement_rtol' is met, at most 50 times;
%                         0 returns the single-precision answer as it is.
%     'refinement_rtol'   Relative residual at which refinement stops.
%                         Default 1e-6.
%
%   Any field of sparta_config in sparta.h may be named as well, such as
%   'nemin', 'imbalance' or 'max_front'. The SPARTA user guide explains each.
%
%   Example:
%     A = delsq(numgrid('S', 202)) + 0.05i * speye(40000);
%     S = SpartaSolver(A);
%     x = S \ ones(40000, 1);
%
%   Always pass the whole matrix, both triangles, even when it is symmetric:
%   refinement multiplies by it.

    properties (SetAccess = private)
        Rows = 0            % number of unknowns
        Kind = ''           % 'symmetric' or 'general'
        FillRatio = 0       % entries in the factors per entry of the matrix
        FactorBytes = 0     % GPU memory the factors occupy, in bytes
        FactorSeconds = 0   % GPU time of the last factorization, in seconds
        PerturbedPivots = 0 % pivots the last factorization nudged off zero
    end

    properties
        RefinementSteps = -1  % see 'refinement_steps' above
        RefinementRtol = 1e-6 % see 'refinement_rtol' above
    end

    properties (Access = private)
        Id = 0              % this solver's number inside sparta_mex
        Matrix = []         % the matrix last factored, for refinement
        AutoKind = true     % whether Kind was detected rather than given
        Settings = struct() % sparta_config fields and preset, for sparta_mex
    end

    methods
        function S = SpartaSolver(A, varargin)
            %SPARTASOLVER  Plan and factor A. See the class help for settings.
            %   A         square sparse matrix, real or complex
            %   varargin  name-value pairs
            kind = S.parseOptions(varargin);
            SpartaSolver.checkMatrix(A, 0);
            S.Rows = size(A, 1);
            S.AutoKind = strcmp(kind, 'auto');
            if S.AutoKind
                kind = SpartaSolver.detectKind(A);
            end
            S.Kind = kind;
            S.plan(A);
            S.factorize(A);
        end

        function factorize(S, A)
            %FACTORIZE  Factor new values, keeping the plan when the pattern allows.
            %   A  square sparse matrix of the size this solver was made for
            SpartaSolver.checkMatrix(A, S.Rows);
            if S.AutoKind && ~strcmp(SpartaSolver.detectKind(A), S.Kind)
                S.Kind = SpartaSolver.detectKind(A);
                S.plan(A);
            end
            try
                [S.FactorSeconds, S.PerturbedPivots] = sparta_mex('factor', S.Id, S.stored(A));
            catch err
                if ~strcmp(err.identifier, 'sparta:pattern')
                    rethrow(err);
                end
                S.plan(A);
                [S.FactorSeconds, S.PerturbedPivots] = sparta_mex('factor', S.Id, S.stored(A));
            end
            S.Matrix = A;
        end

        function x = solve(S, b)
            %SOLVE  Solve A x = b with the current factors.
            %   b  Rows-by-k matrix, one right-hand side per column
            %   x  complex double, the same size as b
            if size(b, 1) ~= S.Rows || ~ismatrix(b)
                error('sparta:invalid', 'b must have %d rows', S.Rows);
            end
            b = full(double(b));
            x = sparta_mex('solve', S.Id, b);
            x = S.refine(b, x);
        end

        function x = mldivide(S, b)
            %MLDIVIDE  x = S \ b, the same as solve(S, b).
            x = S.solve(b);
        end

        function delete(S)
            %DELETE  Release the GPU memory. Runs when the last reference goes.
            if S.Id ~= 0
                try
                    sparta_mex('free', S.Id);
                catch
                end
                S.Id = 0;
            end
        end
    end

    methods (Static)
        function v = version()
            %VERSION  The SPARTA library version, such as '0.3.0'.
            v = sparta_mex('version');
        end

        function d = device()
            %DEVICE  The GPU backend the library was built for: 'metal' or 'vulkan'.
            d = sparta_mex('device');
        end

        function tf = gpuAvailable()
            %GPUAVAILABLE  True when this machine has a GPU SPARTA can use.
            tf = sparta_mex('gpu_available');
        end
    end

    methods (Access = private)
        function kind = parseOptions(S, args)
            % Reads name-value pairs into the solver, returning the 'kind'.
            if mod(numel(args), 2) ~= 0
                error('sparta:invalid', 'settings come in name-value pairs');
            end
            kind = 'auto';
            settings = struct();
            for k = 1:2:numel(args)
                name = lower(args{k});
                value = args{k + 1};
                switch name
                    case 'kind'
                        kind = lower(value);
                        if ~any(strcmp(kind, {'auto', 'symmetric', 'general'}))
                            error('sparta:invalid', 'kind must be auto, symmetric or general');
                        end
                    case 'refinement_steps'
                        S.RefinementSteps = value;
                    case 'refinement_rtol'
                        S.RefinementRtol = value;
                    otherwise
                        settings.(name) = value;
                end
            end
            S.Settings = settings;
        end

        function plan(S, A)
            % Plans A's pattern, replacing any earlier plan.
            if S.Id ~= 0
                sparta_mex('free', S.Id);
                S.Id = 0;
            end
            S.Id = sparta_mex('create', S.stored(A), S.Kind, S.Settings);
            info = sparta_mex('info', S.Id);
            S.FillRatio = info.fill_ratio;
            S.FactorBytes = info.factor_bytes;
        end

        function B = stored(S, A)
            % The entries SPARTA reads: the lower triangle for 'symmetric'.
            B = double(A);
            if strcmp(S.Kind, 'symmetric')
                B = tril(B);
            end
        end

        function x = refine(S, b, x)
            % Corrects x in double precision, column by column, until each
            % residual meets RefinementRtol or stops improving.
            steps = S.RefinementSteps;
            if steps < 0
                steps = 50;
            end
            if steps == 0
                return;
            end
            scale = sqrt(sum(abs(b).^2, 1));
            scale(scale == 0) = 1;
            r = b - S.Matrix * x;
            error_now = sqrt(sum(abs(r).^2, 1)) ./ scale;
            open = error_now > S.RefinementRtol;
            for step = 1:steps
                if ~any(open)
                    break;
                end
                cols = find(open);
                trial = x(:, cols) + sparta_mex('solve', S.Id, r(:, cols));
                residual = b(:, cols) - S.Matrix * trial;
                error_trial = sqrt(sum(abs(residual).^2, 1)) ./ scale(cols);
                better = error_trial < error_now(cols);
                x(:, cols(better)) = trial(:, better);
                r(:, cols(better)) = residual(:, better);
                error_now(cols(better)) = error_trial(better);
                open(cols(~better)) = false;
                open(cols(better)) = error_trial(better) > S.RefinementRtol;
            end
        end
    end

    methods (Static, Access = private)
        function checkMatrix(A, rows)
            % Stops unless A is square, sparse and, if rows > 0, rows by rows.
            if ~issparse(A) || size(A, 1) ~= size(A, 2) || isempty(A)
                error('sparta:invalid', 'A must be a square, nonempty sparse matrix');
            end
            if rows > 0 && size(A, 1) ~= rows
                error('sparta:invalid', 'A has %d rows; this solver was made for %d', size(A, 1), rows);
            end
        end

        function kind = detectKind(A)
            % 'symmetric' when A equals its plain transpose, else 'general'.
            if issymmetric(A)
                kind = 'symmetric';
            else
                kind = 'general';
            end
        end
    end
end
