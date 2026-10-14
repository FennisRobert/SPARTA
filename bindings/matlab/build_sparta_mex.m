function build_sparta_mex(libdir)
%BUILD_SPARTA_MEX  Compile sparta_mex, the gateway SpartaSolver calls.
%
%   build_sparta_mex() builds against the library in target/release of this
%   repository, which `cargo build --release -p sparta-capi` produces.
%   build_sparta_mex(libdir) builds against the library in another folder,
%   such as the lib folder of a packaged SPARTA.
%
%   libdir  folder holding libsparta.dylib (macOS), libsparta.so (Linux), or
%           sparta.dll with sparta.dll.lib (Windows)
%
%   Needs a C compiler that `mex -setup C` accepts. The result lands next to
%   this file; add this folder to the MATLAB path to use SpartaSolver.

here = fileparts(mfilename('fullpath'));
if nargin < 1
    libdir = fullfile(here, '..', '..', 'target', 'release');
end
include = fullfile(here, '..', '..', 'crates', 'sparta-capi', 'include');
source = fullfile(here, 'sparta_mex.c');

if ispc
    % The import library; plain sparta.lib is the static build, which is not
    % what a MEX file should link.
    implib = fullfile(libdir, 'sparta.dll.lib');
    need(implib);
    mex('-R2018a', ['-I' include], source, implib, '-outdir', here);
    % Windows finds a DLL beside the MEX file that loads it.
    copyfile(fullfile(libdir, 'sparta.dll'), here);
else
    if ismac
        library = 'libsparta.dylib';
    else
        library = 'libsparta.so';
    end
    need(fullfile(libdir, library));
    % The rpath lets the MEX file find the library where it was built.
    mex('-R2018a', ['-I' include], ['-L' libdir], '-lsparta', ...
        ['LDFLAGS=$LDFLAGS -Wl,-rpath,' libdir], source, '-outdir', here);
end
fprintf('Built sparta_mex against SPARTA %s (%s).\n', ...
        sparta_mex('version'), sparta_mex('device'));
end

function need(path)
% Stops with build instructions when the library file is missing.
if ~exist(path, 'file')
    error('sparta:build', ['%s not found. Build it from the repository root with\n' ...
          '    cargo build --release -p sparta-capi\n' ...
          'or pass the folder holding it to build_sparta_mex.'], path);
end
end
