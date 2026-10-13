// The look follows the website (website/index.html): its wine red and grey
// tones, Roboto for headings, tables and code, and its dark hero gradient on
// the cover and the chapter openers. Body text is set in EB Garamond. The
// fonts are bundled in fonts/, so build with
//
//   typst compile --font-path documentation/fonts documentation/user_guide.typ

#let red = rgb("#8a0c24")
#let hot = rgb("#ff3e5f")
#let ink = rgb("#121011")
#let body-ink = rgb("#1f1c1d")
#let muted = rgb("#5f5a5b")
#let rule = rgb("#dedad9")
#let band = rgb("#f4f4f5")
#let amber = rgb("#d19a0b")
#let amber-ink = rgb("#7d5a00")
#let amber-bg = rgb("#fbf3d6")
#let sans = "Roboto"
#let serif = "EB Garamond"
#let mono = "Roboto Mono"

#set document(title: "SPARTA User Guide", author: "Robert Fennis and Milan Rother")

// The website's hero: near-black into deep wine, with a red glow on the right.
#let hero(width, height, body) = box(width: width, height: height, {
  place(rect(width: 100%, height: 100%,
    fill: gradient.linear(rgb("#0c0a0b"), rgb("#1a0b0f"), rgb("#2a0a12"), angle: 30deg)))
  place(rect(width: 100%, height: 100%,
    fill: gradient.radial(red.transparentize(50%), red.transparentize(100%), center: (78%, 40%), radius: 55%)))
  body
})

// A lower triangle of random front entries, faded towards its edges: the
// still image of the website's animated multifrontal step. `seed` picks the
// numbers, so each chapter gets its own.
#let front-art(n, seed, size: 9pt, strength: 0.5) = {
  let s = seed
  let rows = ()
  for i in range(n) {
    let cells = ()
    for j in range(i + 1) {
      s = calc.rem(s * 1103515245 + 12345, 2147483648)
      let u = s / 2147483648
      let v = if i == j { 1.2 + 2.6 * u } else { 1.8 * u - 0.9 }
      let a = calc.abs(v)
      let whole = int(a)
      let frac = calc.min(int(calc.round((a - whole) * 10000)), 9999)
      let digits = str(frac)
      let label = (if v < 0 { "-" } else { "\u{2007}" }) + str(whole) + "." + "0" * (4 - digits.len()) + digits
      let d = calc.sqrt(calc.pow((i - n * 0.55) / n, 2) + calc.pow((j - n * 0.35) / n, 2))
      let fade = calc.max(0, 1 - 1.6 * d) * strength
      let colour = if calc.rem(s, 17) == 0 and fade > 0.2 {
        hot.transparentize(100% - fade * 180%)
      } else {
        rgb(222, 212, 214).transparentize(100% - fade * 100%)
      }
      cells.push(text(fill: colour, label))
    }
    rows.push(cells.join(h(1.1em)))
  }
  set text(font: mono, size: size)
  set par(justify: false)
  stack(dir: ttb, spacing: 0.75em, ..rows)
}

// Running header: the logo and the guide's name on the left, the current
// chapter on the right, over a grey rule with a short red lead. Chapter
// opening pages carry the dark band instead, and the cover carries neither.
#let running-header = context {
  let page = here().page()
  let all = query(heading.where(level: 1))
  let starting = all.filter(h => h.location().page() == page)
  if page > 1 and starting.len() == 0 {
    let earlier = all.filter(h => h.location().page() < page)
    let chapter = if earlier.len() > 0 { earlier.last().body } else { [] }
    set text(font: sans, size: 8.5pt, fill: muted)
    grid(columns: (1fr, auto), align: horizon,
      grid(columns: 3, align: horizon, column-gutter: 6pt,
        box(image(height: 1.6em, "logo.svg")),
        text(fill: ink, weight: 500, tracking: 0.12em)[SPARTA],
        text(tracking: 0.1em)[USER GUIDE]),
      text(tracking: 0.1em, upper(chapter)))
    v(-3pt)
    place(line(length: 100%, stroke: 0.5pt + rule))
    line(length: 16mm, stroke: 1.2pt + red)
  }
}
#let running-footer = context {
  if here().page() > 1 {
    set text(font: sans, size: 8pt, fill: muted)
    line(length: 100%, stroke: 0.5pt + rule)
    v(-4pt)
    grid(columns: (1fr, 1fr),
      align(left, text(tracking: 0.08em)[SPARTA · GPU SPARSE DIRECT SOLVER]),
      align(right)[#text(fill: red, weight: 500)[#counter(page).display()] / #counter(page).final().first()])
  }
}
#set page(paper: "a4", margin: (x: 23mm, y: 24mm), header: running-header, footer: running-footer)

#set text(font: serif, size: 11.5pt, lang: "en", fill: body-ink, number-type: "lining")
#show strong: set text(weight: 600)
#set par(justify: true, leading: 0.68em, spacing: 1.1em)
#set heading(numbering: "1.1")
#set list(marker: text(fill: red)[•])
#set enum(numbering: n => text(font: sans, size: 0.85em, weight: 500, fill: red)[#n.])
#show link: set text(fill: red)

// Code: Roboto Mono in the website's palette. A listing sits on the grey band
// with a red rule on the left; inline code gets a faint band of its own.
#set raw(theme: "sparta.tmTheme")
#show raw: set text(font: mono)
#show raw.where(block: true): it => block(width: 100%, fill: band, stroke: (left: 2pt + red),
  inset: (x: 12pt, y: 10pt), text(size: 8.6pt, it))
#show raw.where(block: false): it => box(fill: band, outset: (x: 1.5pt, y: 2.5pt),
  text(size: 0.86em, fill: ink, it))
// A code listing that is never split across pages.
#let code(body) = block(breakable: false, body)

// Tables as on the website: Roboto, grey rules between rows, and a header row
// in small grey capitals over a black rule.
#set table(inset: (x: 7pt, y: 6.5pt), stroke: (x, y) => (bottom: 0.5pt + rule))
#show table: set text(font: sans, size: 9.5pt)
#show table: it => { show raw.where(block: false): set text(size: 1.08em); it }
#show table: set par(justify: false, leading: 0.55em)
#let head(..cells) = table.header(..cells.pos().map(c => table.cell(stroke: (bottom: 1.2pt + ink),
  text(size: 7.5pt, weight: 500, fill: muted, tracking: 0.08em, upper(c)))))

// Callouts. A note is the website's grey band with a red rule; a warning, for
// what is easy to get wrong, is the same in yellow.
#let callout(label, colour, label-colour, background, body) = block(width: 100%, fill: background,
  stroke: (left: 2.5pt + colour), inset: (x: 13pt, y: 11pt), breakable: false, {
  text(font: sans, size: 7.5pt, weight: 500, tracking: 0.12em, fill: label-colour, label)
  v(0.2em)
  body
})
#let note(body) = callout("NOTE", red, red, band, body)
#let warning(body) = callout("WARNING", amber, amber-ink, amber-bg, body)

// Chapters open on a new page under a dark band that runs to the page edges,
// as the hero does on the website: the number in the bright red of its
// animation, the title in white, and a faded front behind them.
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  let number = if it.numbering != none { counter(heading).get().first() } else { none }
  place(top + left, dx: -23mm, dy: -24mm, hero(210mm, 54mm, {
    place(top + right, dx: -6mm, dy: 4mm,
      rotate(-8deg, reflow: false, skew(ax: -12deg, front-art(7, 977 + 31 * if number == none { 0 } else { number }, size: 8.5pt, strength: 0.42))))
    place(bottom + left, dx: 23mm, dy: -9mm, block(width: 150mm, {
      set par(justify: false, leading: 0.4em)
      if number != none {
        text(font: sans, size: 11pt, weight: 500, tracking: 0.14em, fill: hot)[CHAPTER #number]
        v(0.5em)
      }
      text(font: sans, size: 23pt, weight: 300, tracking: 0.08em, fill: white, upper(it.body))
    }))
  }))
  v(30mm + 1.2em)
}
// Headings in spaced Roboto capitals. Sections: the number in red, the title
// in black.
#show heading.where(level: 2): it => block(above: 1.8em, below: 0.9em, sticky: true, {
  set text(font: sans, size: 12pt, weight: 500, tracking: 0.09em, fill: ink)
  set par(justify: false)
  if it.numbering != none { text(fill: red, counter(heading).display()); h(0.8em) }
  upper(it.body)
})
#show heading.where(level: 3): it => block(above: 1.4em, below: 0.7em, sticky: true, {
  set text(font: sans, size: 9.5pt, weight: 500, tracking: 0.1em, fill: ink)
  set par(justify: false)
  if it.numbering != none { text(fill: muted, counter(heading).display()); h(0.8em) }
  upper(it.body)
})

// Contents in Roboto, chapters a step heavier with a little air above.
#set outline(indent: auto)
#set outline.entry(fill: repeat(text(fill: rule)[.], gap: 3pt))
#show outline.entry: set text(font: sans, size: 9.5pt, fill: body-ink)
#show outline.entry: set block(spacing: 0.62em)
#show outline.entry.where(level: 1): it => {
  v(0.75em, weak: true)
  text(weight: 500, fill: ink, tracking: 0.08em, size: 8.8pt, upper(it))
}

// Cover: the website's hero over the whole page, with the faded front on the
// right and the facts strip along the bottom.
#page(margin: 0pt, header: none, footer: none, hero(100%, 100%, {
  place(top + left, dx: 118mm, dy: 118mm,
    rotate(-10deg, reflow: false, skew(ax: -14deg, front-art(10, 20260929, size: 10pt, strength: 0.6))))
  place(top + left, dx: 23mm, dy: 16mm, {
    set text(font: sans, size: 8.5pt, fill: rgb("#bdb6b7"))
    box(image(height: 2.4em, "logo.svg"))
    h(8pt)
    box(baseline: -0.75em, text(fill: white, weight: 500, tracking: 0.14em)[SPARTA])
    h(6pt)
    box(baseline: -0.75em, text(tracking: 0.08em)[· GPU SPARSE DIRECT SOLVER])
  })
  place(top + left, dx: 23mm, dy: 92mm, block(width: 120mm, {
    set par(justify: false)
    text(font: sans, size: 9pt, tracking: 0.12em, fill: rgb("#b7aeaf"))[SPARSE ACCELERATED RAPID TRIANGULAR ANALYSER]
    v(1.4em)
    text(font: sans, size: 56pt, weight: 500, tracking: 0.08em, fill: white)[SPARTA]
    v(0.1em)
    text(font: sans, size: 20pt, weight: 300, tracking: 0.16em, fill: rgb("#e6e0e1"))[USER GUIDE]
    v(1.6em)
    line(length: 22mm, stroke: 1.5pt + hot)
    v(1.2em)
    text(font: serif, style: "italic", size: 14.5pt, fill: rgb("#e6e0e1"))[Supernodal multifrontal LDL#super[T] and LU
      for large sparse systems, with every numeric kernel on the GPU.]
  }))
  place(bottom + left, dx: 23mm, dy: -18mm, block(width: 164mm, {
    set text(font: sans, size: 8.5pt, fill: rgb("#c9c1c2"))
    set par(justify: false, leading: 0.5em)
    set align(top + left)
    line(length: 100%, stroke: 0.5pt + white.transparentize(82%))
    v(2pt)
    let fact(title, body) = block(inset: (top: 8pt, right: 8pt), {
      text(size: 8.5pt, weight: 500, tracking: 0.08em, fill: white, upper(title))
      linebreak()
      body
    })
    grid(columns: (1fr,) * 4, column-gutter: 0pt,
      stroke: (x, y) => if x > 0 { (left: 0.5pt + white.transparentize(82%)) },
      inset: (x, y) => if x > 0 { (left: 10pt) } else { 0pt },
      fact[LDL#super[T] and LU][supernodal multifrontal, GPU-resident],
      fact[complex64 + complex128][mixed-precision refinement],
      fact[Nested dissection][SCALPEL, a parallel ordering],
      fact[Rust, Python, C/C++][native interfaces])
    v(16pt)
    text(size: 9pt, tracking: 0.12em, fill: white)[ROBERT FENNIS #h(0.6em) · #h(0.6em) MILAN ROTHER]
  }))
}))

#outline(title: [Contents], depth: 2)
#pagebreak()

= Introduction

This guide takes you from installing SPARTA to solving your own systems
accurately. It assumes you can program in Rust or Python, but not that you
know how a sparse direct solver works: the ideas are explained as they come up.

== What SPARTA does

SPARTA solves systems of linear equations,

$ A x = b, $

where $A$ is a large square matrix, $b$ is a known vector (the *right-hand
side*) and $x$ is the unknown vector you want. Such systems come out of
simulations: a finite-element model of an antenna, a circuit or a structure
produces one matrix per frequency or time step.

These matrices are *sparse*: nearly all their entries are zero. A matrix with a
million rows might have only ten or twenty nonzeros per row. SPARTA stores and
works on the nonzeros only, which is what makes such large systems solvable at
all.

SPARTA is a *direct* solver. It computes the answer in a fixed number of steps,
as opposed to an iterative solver, which improves a guess until it is good
enough and may converge slowly or not at all on difficult systems. The price of
a direct solver is memory and work up front; SPARTA pays that price on the GPU
of an Apple Silicon Mac, where it is paid quickly.

== How a direct solver works

A direct solver does what you would do by hand with Gaussian elimination, but
records the result so it can be reused. It splits the work into three steps.

+ *Analyze.* Look only at _where_ the nonzeros are (the *pattern*), not at
  their values. Choose an order in which to eliminate the unknowns, and plan
  the memory and the computation. This runs on the CPU.
+ *Factor.* Use the values to split $A$ into simpler matrices, its *factors*,
  whose product is $A$. This is the expensive step, and it runs on the GPU.
+ *Solve.* Use the factors to find $x$ for a given $b$. This is cheap compared
  with factoring.

The split pays off because each step can be repeated on its own:

- *A new right-hand side* needs only a new solve.
- *New values in the same pattern*, such as the next frequency of a sweep, need
  a new factorization, but reuse the analysis and the memory it reserved.
- *A new pattern* needs a new analysis. In Python, `factorize` runs it for you
  when the matrix it is given has a different pattern.

Two things decide how much work and memory a factorization takes.

*Fill-in.* Eliminating an unknown creates new nonzeros in the factors where
$A$ had zeros. The factors are therefore denser than $A$ itself, often by ten
times or more.

*Ordering.* How much fill-in appears depends heavily on the order in which the
unknowns are eliminated. Finding a good order is the main job of the analysis.
SPARTA uses *nested dissection*: it repeatedly cuts the problem into two halves
joined by a small *separator*, and eliminates the halves before the separator.

SPARTA gathers the work into dense blocks called *fronts*, because dense blocks
are what a GPU computes fastest. A group of columns handled together in one
front is a *supernode*.

== Two factorization methods

The kind of matrix decides which factorization SPARTA can use.

#table(columns: (auto, 1fr, auto),
  head([Method], [Used for], [Relative cost]),
  [$L D L^T$], [*Symmetric* matrices, real or complex: $A$ equals its own
    transpose, $A = A^T$. Only one triangle needs storing.], [1×],
  [$L U$], [*Every other square matrix*: unsymmetric, only the pattern
    symmetric, or Hermitian ($A$ equal to its conjugate transpose).], [about 2×],
)

#warning[*Symmetric means equal to the transpose, not the conjugate transpose.*
A complex Hermitian matrix is solved with LU. A symmetric pattern with
unsymmetric values also needs LU.]

While factoring, each step divides by one entry, the *pivot*. A pivot that is
nearly zero amplifies rounding errors. Both methods therefore reorder rows or
use small $2 times 2$ blocks locally to avoid small pivots, which is called
*pivoting*.

== Precision in one paragraph

The GPU computes in *single precision*, which holds about seven significant
digits. That makes it fast, but on a typical simulation matrix the answer
comes out with a *relative residual* of about $10^(-5)$, where the residual
$b - A x$ measures how well $x$ satisfies the equations. SPARTA can recover
double-precision accuracy by *iterative refinement*, or compute entirely in
double precision on the CPU. @precision explains exactly how.

== What is supported

#table(columns: (auto, 1fr),
  head([], [Support]),
  [GPU], [Apple Silicon (M1 or later) through Metal. AMD and NVIDIA GPUs on
    Windows and Linux through Vulkan, in alpha: correct, not yet tuned.],
  [CPU], [A double-precision backend on any 64-bit machine, used when you ask
    for it.],
  [Languages], [Rust and Python, covered in this guide. A C and C++ interface
    exists as well; see `crates/sparta-capi/README.md`.],
  [Matrices], [Square, real or complex, stored in compressed sparse column
    form (Rust) or any SciPy sparse format (Python).],
  [Ordering], [SCALPEL, SPARTA's own parallel nested dissection.],
)

#pagebreak()
== Key terms

#table(columns: (auto, 1fr),
  head([Term], [Meaning]),
  [Sparse matrix], [A matrix whose entries are almost all zero, stored as its nonzeros only.],
  [Pattern], [The positions of the nonzeros, regardless of their values.],
  [Right-hand side], [The known vector $b$ in $A x = b$.],
  [Factorization], [Splitting $A$ into factors that make solving easy.],
  [Fill-in], [Nonzeros the factors have where $A$ had zeros.],
  [Ordering], [The order in which unknowns are eliminated. Decides the fill-in.],
  [Front], [A dense block of the factorization, computed as one unit.],
  [Supernode], [A group of columns factored together in one front.],
  [Pivot], [The entry divided by at each elimination step.],
  [Residual], [$b - A x$. Its size relative to $b$ shows how well $x$ solves the system.],
  [Refinement], [Correcting an answer using its residual, to gain accuracy.],
  [CSC], [Compressed sparse column: the storage format SPARTA reads (@csc).],
)

= Installation

== Requirements per operating system

*macOS.* An Apple Silicon Mac (M1 or later) running macOS 13 or later runs
everything, including the GPU. An Intel Mac runs the analysis and the
double-precision CPU backend, but has no GPU.

*Windows.* Any 64-bit machine runs the analysis and the CPU backend. AMD and
NVIDIA GPUs run through the Vulkan backend, which is in alpha: tested on an
NVIDIA RTX 3080 and an AMD Radeon integrated GPU, correct but not yet tuned
for speed. Building it needs `glslc` from the Vulkan SDK and the `vulkan`
feature (`cargo build --release --features vulkan`); see "Vulkan" in the
README. For a fast CPU backend, install a BLAS library (@blas).

*Linux.* The same as Windows: the analysis, the CPU backend and the alpha
Vulkan backend. Not yet tested on Linux.

== Packages you need

#table(columns: (auto, 1fr),
  head([Package], [What it is for]),
  [Rust toolchain (`rustc`, `cargo`)], [Compiling SPARTA. Needed for every
    install route except a prebuilt Python wheel.],
  [Xcode command line tools], [macOS only: compiling the Metal bridge.],
  [git], [Downloading SPARTA from GitHub.],
  [Python 3.9 or later, with pip], [The Python package.],
  [NumPy and SciPy], [Used by the Python package. pip installs them automatically.],
  [OpenBLAS or Intel MKL], [Optional, Linux and Windows only: makes the CPU backend fast.],
)

== Installing the packages

*macOS*

#code(```sh
# Metal bridge tools and git
xcode-select --install
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```)

For Python, use the installation you already run your simulations with
(python.org, Homebrew or conda).

*Linux (Debian, Ubuntu)*

#code(```sh
sudo apt install build-essential git python3 python3-pip python3-venv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```)

On Fedora or RHEL, the first line is `sudo dnf install gcc git python3 python3-pip`.

*Windows*

+ Rust: download and run `rustup-init.exe` from #link("https://rustup.rs").
  When it asks, let it install the Visual Studio C++ Build Tools, which Rust
  needs to link.
+ git: install from #link("https://git-scm.com/download/win").
+ Python: install from #link("https://www.python.org/downloads/").

*Check* that everything is found by opening a new terminal and running:

#code(```sh
cargo --version
git --version
python3 --version     # "python --version" on Windows
```)

== Installing SPARTA

SPARTA is not published on crates.io or PyPI. Each language has the same
three routes:

+ *You have access to the GitHub repository.*
+ *You have a copy of the source, but no GitHub access.*
+ *You have prebuilt binaries*, so nothing needs compiling.

Routes 1 and 2 compile SPARTA, which takes a few minutes the first time.

=== Rust

The package is called `sparta-solver` and is imported as `sparta`.

*1. With GitHub access.* Add to your project's `Cargo.toml`:

#code(```toml
[dependencies]
sparta-solver = { git = "ssh://git@github.com/FennisRobert/SPARTA.git" }
```)

This needs an SSH key registered with your GitHub account. Cargo fetches
private repositories most reliably through your own git, so also add this to
`~/.cargo/config.toml`:

#code(```toml
[net]
git-fetch-with-cli = true
```)

*2. From a copy of the source.* Point Cargo at the folder that contains
SPARTA's `Cargo.toml`:

#code(```toml
[dependencies]
sparta-solver = { path = "../SPARTA" }
```)

*3. Prebuilt binaries.* Not applicable: Cargo always compiles Rust libraries
from source. If you received a source archive, unpack it and use route 2.

=== Python

Install into the Python environment you run your simulations in. The package
is imported as `sparta`. Routes 1 and 2 need the Rust toolchain installed
first, because pip compiles SPARTA's Rust code during the install.

*1. With GitHub access.*

#code(```sh
git clone git@github.com:FennisRobert/SPARTA.git
cd SPARTA
pip install .
```)

Or in one line, without keeping a copy of the source:

#code(```sh
pip install "git+ssh://git@github.com/FennisRobert/SPARTA.git"
```)

*2. From a copy of the source.* Unpack it, then install from its top folder,
the one containing `pyproject.toml`:

#code(```sh
cd path/to/SPARTA
pip install .
```)

*3. From a prebuilt wheel.* A wheel is a ready-compiled package that needs no
Rust and no compiler. One wheel works for every Python from 3.9 up, on Apple
Silicon only:

#code(```sh
pip install sparta_solver-0.1.0-cp39-abi3-macosx_11_0_arm64.whl
```)

== Checking the installation

*Rust:* build the example in @quick-rust with `cargo run --release`. It prints
the solution of a small system.

*Python:*

#code(```python
import sparta
print(sparta.__version__)
print(sparta.gpu_available())   # True on an Apple Silicon Mac
```)

If `gpu_available()` prints `False` on an Apple Silicon Mac, the process
cannot see the GPU. This happens in some sandboxed or remote sessions; run
from a normal Terminal window.

== A fast CPU backend on Linux and Windows <blas>

This only matters if you use the double-precision CPU backend off a Mac. That
backend hands its large dense operations to a BLAS library, chosen when SPARTA
is compiled. Without one it still gives correct answers, just much more
slowly. On a Mac, Apple's Accelerate is always used and nothing needs doing.

Set two environment variables while installing:

#table(columns: (auto, 1fr),
  head([Variable], [Meaning]),
  [`SPARTA_BLAS_LIB`], [Library to link, without the `lib` prefix: `openblas`, or `mkl_rt` for Intel MKL.],
  [`SPARTA_BLAS_DIR`], [Folder containing it, if it is not on the default search path.],
)

#code(```sh
# OpenBLAS
sudo apt install libopenblas-dev
SPARTA_BLAS_LIB=openblas pip install .          # or: cargo build --release

# Intel MKL (preferred on x86 processors)
source /opt/intel/oneapi/setvars.sh
SPARTA_BLAS_DIR=$MKLROOT/lib SPARTA_BLAS_LIB=mkl_rt pip install .
```)

On Windows, install the `mkl` and `mkl-devel` packages from PyPI into a virtual
environment, here at `C:\venv`, and point the build at its `Library` folder:

#code(```sh
SPARTA_BLAS_DIR='C:\venv\Library\lib' SPARTA_BLAS_LIB=mkl_rt pip install .
```)

`C:\venv\Library\bin` must be on `PATH` whenever SPARTA runs.

MKL is preferred on x86 because SPARTA spreads its own work across threads and
needs to control BLAS threading per thread, which MKL allows and OpenBLAS does
not.

= Quick use

This chapter solves a small system with every setting left at its default.
The next chapter explains the settings.

== How the matrix is passed <csc>

*Python* takes any SciPy sparse matrix and converts it for you.

*Rust* takes the matrix in *compressed sparse column* (CSC) form: three arrays
that list the nonzeros column by column.

#table(columns: (auto, 1fr),
  head([Array], [Contents]),
  [`col_ptr`], [$n + 1$ offsets. The nonzeros of column $j$ are entries `col_ptr[j]` up to, but not including, `col_ptr[j + 1]` of the other two arrays.],
  [`row_idx`], [The row of each nonzero, counting from zero, ascending and without repeats within a column.],
  [`values`], [The value of each nonzero, in the same order.],
)

For example, the symmetric matrix

$ A = mat(4, 1, 0; 1, 3, 1; 0, 1, 2) $

has this lower triangle in CSC form:

#code(```rust
// Column 0 is entries 0-1, column 1 is entries 2-3, column 2 is entry 4.
let col_ptr = [0, 2, 4, 5];
let row_idx = [0, 1, 1, 2, 2];
let values  = [4.0, 1.0, 3.0, 1.0, 2.0];
```)

For a symmetric matrix, pass only the lower triangle: the entries on and
below the diagonal. For any other matrix, pass every nonzero.

== Solving in Rust <quick-rust>

#code(```rust
use sparta::{Analysis, Complex32, Config, Solver};

fn main() -> sparta::Result<()> {
    // A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]], symmetric: lower triangle only.
    let col_ptr = [0, 2, 4, 5];
    let row_idx = [0, 1, 1, 2, 2];
    let values = [4.0, 1.0, 3.0, 1.0, 2.0].map(|v| Complex32::new(v, 0.0));
    let b = [1.0, 2.0, 3.0].map(|v| Complex32::new(v, 0.0));

    // 1. Analyze the pattern.
    let analysis = Analysis::ldlt(3, &col_ptr, &row_idx, &Config::default())?;
    let mut solver = Solver::new(analysis)?;
    // 2. Factor the values.
    solver.factor(&values)?;
    // 3. Solve for x.
    let x = solver.solve(&b)?;
    println!("{x:?}");
    // 4. Free the GPU memory.
    solver.destroy();
    Ok(())
}
```)

`Analysis::ldlt` is for symmetric matrices. For any other matrix, use
`Analysis::lu` and pass the whole matrix. Values are complex numbers with
single-precision parts (`Complex32`); give a real matrix an imaginary part of
zero.

== Solving in Python

#code(```python
import numpy as np
from scipy.sparse import csc_matrix
import sparta

A = csc_matrix([[4.0, 1.0, 0.0],
                [1.0, 3.0, 1.0],
                [0.0, 1.0, 2.0]])
b = np.array([1.0, 2.0, 3.0])

solver = sparta.analyze(A)   # 1. analyze
solver.factorize()           # 2. factor
x = solver.solve(b)          # 3. solve
solver.destroy()             # 4. free
```)

Python looks at the matrix and chooses the method itself: LDLᵀ if it is
symmetric, LU otherwise. The result is a complex NumPy array, also for real
input. Python also refines the answer to double-precision accuracy by
default (@precision).

== Solving again

The three steps can be repeated separately.

*A new right-hand side:* call `solve` again. Several at once go faster
together:

#code(```rust
// Rust: the columns end to end; column j is b[j*n..(j+1)*n].
let x = solver.solve_many(&b, nrhs)?;
```)
#code(```python
# Python: B has shape (n, nrhs), one right-hand side per column.
X = solver.solve_many(B)
```)

*New values with the same pattern*, such as the next frequency: factor again.
The analysis and the memory are reused.

#code(```rust
// Rust: the values in the same CSC order as the first time.
solver.factor(&new_values)?;
```)
#code(```python
# Python: same pattern, new values.
solver.factorize(matrix=A_next)
```)

*A new pattern:* start again from the analysis. In Python, `factorize` does
this itself: given a matrix with a different pattern, or a different size, it
analyzes again with the same settings and then factors. `solver.reanalyses`
counts how often that happened. Each one costs a full analysis and new factor
memory, so a count that keeps rising in a loop means the pattern is not stable
between calls, for example because explicit zeros come and go.

== Freeing memory

A factorization of a large system can hold more than a gigabyte of GPU memory
until its solver is freed. Call `destroy()` when you are done, as in the
examples above. In Rust, dropping the solver also frees it. In Python the
memory otherwise stays claimed for as long as anything refers to the solver,
so prefer `destroy()` or a `with` block:

#code(```python
with sparta.analyze(A) as solver:
    solver.factorize()
    x = solver.solve(b)
```)

= Settings and configuration

The defaults are measured and suit most systems. This chapter explains every
setting, and how to set them from Rust and Python.

== How settings work

Settings belong to one of two moments:

- *Analysis settings* shape the plan: the ordering, the size of fronts, the
  GPU kernel shapes. They are fixed once the matrix is analyzed. To change
  one, analyze again.
- *Solve settings* (Python only) control refinement and checking. They can be
  changed for every factorization.

Every setting has the same name in Rust and Python, with one exception: in
Rust, the four SCALPEL settings live inside a nested `scalpel` field.

== Telling SPARTA about your matrix

=== In Rust

You choose the method by the constructor you call, and store the matrix to
match. Rust does not check symmetry, because it only ever sees one triangle.

#table(columns: (auto, auto, 1fr),
  head([Constructor], [Pass], [For]),
  [`Analysis::ldlt(n, &col_ptr, &row_idx, &config)`], [lower triangle], [symmetric matrices],
  [`Analysis::lu(n, &col_ptr, &row_idx, &config)`], [whole matrix], [every other square matrix],
)

=== In Python

By default Python inspects the matrix and picks LDLᵀ if it equals its
transpose, LU otherwise. Three arguments to `sparta.analyze` change that:

#table(columns: (auto, auto, 1fr),
  head([Argument], [Default], [Effect]),
  [`mtype`], [`MatrixType.AUTO`], [Declares what kind of matrix this is. SPARTA checks the declaration: if you declare a matrix symmetric and it is not, `analyze` raises `ValueError`.],
  [`path`], [`Path.AUTO`], [Forces a method: `Path.LDLT` or `Path.LU`.],
  [`triangle`], [`"full"`], [`"lower"` if you pass only the lower triangle of a symmetric matrix. LU and Hermitian input need `"full"`.],
)

The symmetry check allows a small absolute difference between mirrored
entries, `sparta.EPS`, which starts at `1e-10`. Set it before calling
`analyze`; `sparta.EPS = 0` demands exact symmetry. With a larger tolerance,
LDLᵀ uses the lower triangle as it is, not an average of the two triangles.

== Configuring in Rust

All analysis settings live in one `Config` struct, which you pass to
`Analysis::ldlt` or `Analysis::lu`. Start from the defaults or a preset and
change only what you need:

#code(```rust
use sparta::{Config, OrderingMethod, Preset};

let mut config = Config {
    ordering: OrderingMethod::Scalpel,
    nemin: 32,
    ..Config::default()
};
config.scalpel.wide_cycles = 6;          // SCALPEL settings are nested

let config = Preset::Sweep.config();     // or start from a preset
```)

The analysis keeps a copy of its `Config`. To change a setting later, build a
new `Analysis` and `Solver`.

#table(columns: (auto, 1fr),
  head([Enum], [Members]),
  [`OrderingMethod`], [`Scalpel`],
  [`Preset`], [`Balanced`, `Sweep`, `OneShot`, `StructuredGrid`, `Careful` (@presets)],
)

== Configuring in Python

Pass settings as keyword arguments to `sparta.analyze`:

#code(```python
solver = sparta.analyze(A, ordering=sparta.Ordering.SCALPEL, nemin=32)
```)

To keep a whole set of settings together, use `sparta.Config`. It is
immutable: `with_` returns a changed copy.

#code(```python
config = sparta.Config.preset("sweep").with_(nemin=32)
solver = sparta.analyze(A, config=config)
```)

#table(columns: (auto, 1fr),
  head([Call], [Returns]),
  [`sparta.Config.defaults()`], [The default settings, read from the compiled library.],
  [`sparta.Config.preset(name)`], [A preset by name, such as `"sweep"`.],
  [`config.with_(**changes)`], [A copy with the named settings changed.],
  [`config.to_dict()`], [The settings as keyword arguments for `analyze`.],
  [`sparta.presets()`], [Every preset name with a one-line summary.],
)

*Which value wins.* A keyword argument overrides the same setting in
`config`. One exception: `max_front` has a default of its own, `None`, and
passing it does not override the config. To remove a config's front limit,
change the config itself:

#code(```python
config = config.with_(max_front=None)
```)

In a `Config`, `ordering` is stored as text (`"scalpel"`), so use the enum's
`.value` there, and the enum itself as a keyword argument.

*Changing settings later.* The solve settings and `force_cpu` can be passed to
`factorize` to change them for an existing solver; anything left out keeps its
current value. Everything else needs a new `analyze`.

#code(```python
solver.factorize(refinement_rtol=1e-10)   # tighter refinement from now on
```)

=== Python enums

#table(columns: (auto, 1fr, 1fr),
  head([Enum], [Members], [Meaning]),
  [`Ordering`], [`SCALPEL`], [Which ordering to use. SCALPEL is the only one.],
  [`Path`], [`AUTO`, `LDLT`, `LU`], [Which method to use. `AUTO` decides from the matrix type.],
  [`Backend`], [`GPU`, `CPU`], [Where a tuned configuration runs (@optimize).],
  [`MatrixType`], [`AUTO`], [Look at the matrix and decide: LDLᵀ if symmetric, else LU.],
  [], [`REAL_SYMMETRIC`, `COMPLEX_SYMMETRIC`], [$A = A^T$. Uses LDLᵀ.],
  [], [`REAL_UNSYMMETRIC`, `COMPLEX_UNSYMMETRIC`], [Any square matrix. Uses LU.],
  [], [`REAL_STRUCTURALLY_SYMMETRIC`, `COMPLEX_STRUCTURALLY_SYMMETRIC`], [Symmetric pattern, unsymmetric values. Uses LU.],
  [], [`HERMITIAN`], [$A$ equals its conjugate transpose. Uses LU.],
  [`RefinementStatus`], [`CONVERGED`, `STAGNATED`, `ITERATION_LIMIT`, `NONFINITE`], [How refinement ended (@precision).],
)

The enums also contain members reserved for later versions. Selecting one
raises `NotImplementedError` rather than silently doing something else.

== Presets <presets>

A preset is a named starting point for a kind of workload. It is not a
guarantee: the best settings depend on your matrices, so measure before you
keep a change.

#table(columns: (auto, auto, 1fr),
  head([Rust], [Python], [Use it for, and what it changes]),
  [`Balanced`], [`"balanced"`], [The defaults. One factorization of an unstructured mesh.],
  [`Sweep`], [`"sweep"`], [One pattern factored many times. `wide_cycles = 6` and `separator_trials = 8`: more ordering work, paid once.],
  [`OneShot`], [`"one-shot"`], [Factor once and discard. `wide_cycles = 0` and `separator_trials = 1`: a fast ordering with more fill-in.],
  [`StructuredGrid`], [`"structured-grid"`], [Regular grids. `nemin = 32`: wider fronts.],
  [`Careful`], [`"careful"`], [Ill-conditioned systems. Strict LU pivoting (`lu_diag_threshold = 0`) and `imbalance = 0.6`: a more balanced split.],
)

== Tuning to your computer <optimize>

The defaults were measured on one machine, and presets are educated guesses.
`sparta.optimize` measures instead. It factors your matrices under many
candidate settings, keeps whatever is clearly faster and no less accurate,
and saves the result under a name you choose.

#code(```python
result = sparta.optimize([A1, (A2, b2)], name="antenna")
solver = sparta.analyze(A, config="antenna")      # any later session
```)

Pass matrices like the ones you will factor, at representative sizes: the
best settings follow the matrices as much as the hardware. A problem is a
sparse matrix, or a `(matrix, rhs)` pair with a vector or an `(n, k)` block.
Without problems, `sparta.optimize()` tunes on two standard finite-element
Helmholtz systems of 14 and 47 thousand unknowns, which is a reasonable
general tuning for the computer.

*What it prints.* Every measurement prints one line as it finishes, with the
count done, the total, the time spent and an estimate of the time left:

#code(```text
[ 5/78 0:06 | ~1:29 left] GPU nemin=16                  99.1 ms  (current)
[ 6/78 0:07 | ~1:29 left] GPU nemin=8                  101.8 ms  (+2.7% vs current)
...
SPARTA optimize: 2 problem(s) on Apple M4 Pro, 1:27
            default       tuned     gain
  GPU      100.4 ms     97.9 ms    1.03x
  CPU      338.7 ms    330.7 ms    1.02x
  Result: uses the GPU, 1.03x the speed of the default configuration (2% less time).
  Changed: narrow_cols 32 -> 16
```)

The summary compares the default and the tuned settings, measured back to
back at the end. Ctrl-C stops the search.

#table(columns: (auto, auto, 1fr),
  head([Argument], [Default], [What it does]),
  [`problems`], [standard set], [The matrices to tune on.],
  [`name`], [none], [Save the result under this name. Preset names are refused.],
  [`gpu`, `cpu`], [`True`, `True`], [Which backends to tune. With both, the result uses the faster one, and each keeps its own tuned settings, so `factorize(force_cpu=...)` can still switch.],
  [`start`], [defaults], [A `Config`, profile or preset to start from and compare against.],
  [`analysis`], [`False`], [Count analysis time too. Turn on when each pattern is factored only once.],
  [`time_limit`], [none], [Seconds after which the search stops and keeps the best so far.],
  [`iterations`], [`1`], [How many full passes over all settings to make. Each pass starts again from the settings the last one found (@optimize-iterations).],
  [`min_gain`], [`0.03`], [The smallest improvement worth keeping, as a fraction. `0.03` keeps a change only when it saves more than 3% of the time.],
  [`verbose`], [`True`], [Print progress and the summary.],
  [`progress`], [none], [A function called with each progress report, for your own display.],
  [`mtype`, `path`, `triangle`], [as `analyze`], [How to read the matrices.],
)

*How it searches.* One setting at a time: first the ordering and front
settings, on whichever backend is faster, then each backend's own settings.
Each candidate is timed as the median of several factorize-and-solve runs.
`lu_diag_threshold` is never changed, because it trades accuracy for speed.
A full run on the standard problems takes one to two minutes on an M4 Pro.

*What counts as faster.* Timings vary a little from run to run, and a search
that follows every small difference ends up following noise. A change is
therefore kept only when all of these hold:

+ It saves more than `min_gain` of the time, 3% by default.
+ It saves more than the timing noise, measured as the spread of that
  candidate's own runs. On a small matrix that factors in a few
  milliseconds, the noise can be 10% or more, and then the bar is that high.
+ It is still faster when the old and new values are timed again, head to
  head. A candidate that was only lucky once prints
  `was not faster on a recheck` and is dropped.
+ Its residual is at most ten times the default settings' residual.

Raise `min_gain` to change only what clearly matters, for example
`min_gain=0.1` for 10%.

The result, `sparta.Optimization`, has `config`, `backend`, `speedup`,
`seconds` (default and tuned, per backend), `changes`, `iterations` and
every `trials` entry. `print(result)` gives the summary.

=== Iterations <optimize-iterations>

Settings affect each other. A wider `nemin` changes which fronts exist, and
so which kernel geometry suits them. A single pass tunes each setting while
the ones after it still have their old values. With `iterations=3` the
search makes up to three passes, each starting from what the last one found,
so every setting is tuned again once the others have moved.

#code(```python
result = sparta.optimize([A1, A2], name="antenna", iterations=3)
```)

A pass that changes nothing means the settings have settled, and the search
stops there without spending time on the remaining passes:

#code(```text
[ 66/286 0:15 | ~0:48 left] iteration 2 of 3: starting again from the settings found so far
[126/130 0:28 | ~0:01 left] iteration 2 changed nothing: the settings have settled, skipping the rest
...
  Iterations: 2 of 3, stopped early.
```)

So a large value costs nothing once the settings have settled. Each pass
takes about as long as the first. The progress total and the time left count
all requested passes, and drop when the search stops early.
`result.iterations` gives `(run, asked)`.

=== Saved profiles

A profile is a saved `Config`. All profiles live in one file, `configs.json`,
in the SPARTA folder of your home directory: `~/Library/SPARTA` on macOS,
`~/.sparta` elsewhere. Set `SPARTA_CONFIG` to use another file.

#table(columns: (auto, 1fr),
  head([Call], [What it does]),
  [`sparta.analyze(A, config="name")`], [Use a saved profile, or a preset of that name.],
  [`sparta.Config.load(name)`], [The saved profile as a `Config`.],
  [`config.save(name, note="")`], [Save any `Config` as a profile.],
  [`sparta.Config.delete(name)`], [Remove a profile.],
  [`sparta.profiles()`], [Every profile with a one-line summary: chip, backend, gain and date.],
  [`sparta.Config.store()`], [The path of `configs.json`.],
)

A profile records the chip it was tuned on. Loading it on a different chip
works, with a warning to tune again. A profile saved by an older SPARTA
loads too: settings added since then take their defaults.

In Rust, `sparta::optimize::optimize` runs the same search, and
`sparta::profiles` reads and writes the same file:

#code(```rust
use sparta::optimize::{optimize, Options, Problem};

let problems = vec![Problem::ldlt(n, ptr, rows, values)?];
let options = Options { iterations: 3, ..Options::default() };
let outcome = optimize(&problems, &options, &mut |p| {
    eprintln!("{p}");   // one progress line
    true                // false stops the search
})?;
println!("{outcome}");
sparta::profiles::save("antenna", &outcome.profile())?;
let config = sparta::profiles::load("antenna")?.config;
```)

`cargo run --release --example optimize -- NAME` tunes on the standard
problems and saves the result as `NAME`.

== All settings

Types are given in a language-neutral form. `uint32` is `u32` in Rust and
`int` in Python; `uint64` is `usize` and `int`; `float64` is `f64` and
`float`; "optional" is `Option` in Rust and `None` in Python.

#table(columns: (auto, auto, auto, 1fr),
  head([Name], [Type], [Default], [What it is for]),
  table.cell(colspan: 4)[*Solve settings* (Python only)],
  [`refinement_steps`], [int], [`-1`], [Maximum refinement corrections per right-hand side.],
  [`refinement_rtol`], [float64], [`1e-6`], [Residual at which refinement stops.],
  [`refinement_minimize`], [bool], [`True`], [Scale each correction to shrink the residual most.],
  [`equilibrate`], [bool], [`False`], [Balance row and column sizes before converting to single precision.],
  [`safe_mode`], [bool], [`False`], [Re-check every new matrix passed to `factorize`.],
  [`verbose`], [bool], [`False`], [Print a short summary per solve.],
  [`mtype`, `path`, `triangle`], [enum, enum, text], [automatic, `"full"`], [How to read and factor the matrix.],
  table.cell(colspan: 4)[*Ordering*],
  [`ordering`], [enum], [SCALPEL], [The nested dissection. SCALPEL is the only one.],
  [`imbalance`], [float64], [`0.8`], [SCALPEL only. How unequal a split may be.],
  [`wide_cycles`], [uint64], [`4`], [SCALPEL only. Separator refinement cycles per split.],
  [`fm_passes`], [uint64], [`2`], [SCALPEL only. Refinement passes per separator.],
  [`separator_trials`], [uint64], [`4`], [SCALPEL only. Candidate separators tried per split.],
  table.cell(colspan: 4)[*Fronts*],
  [`nemin`], [uint64], [`16`], [How eagerly columns merge into wider fronts.],
  [`max_front`], [uint64, optional], [unlimited], [Largest front allowed, in rows.],
  [`small_front_rows`], [uint32], [`32`], [Front height where the GPU leaves its serial kernel.],
  table.cell(colspan: 4)[*Numerics*],
  [`force_cpu`], [bool], [`False`], [Double precision on the CPU instead of the GPU.],
  [`lu_diag_threshold`], [float64], [`0.0`], [LU only. How strongly to prefer the diagonal as pivot.],
  table.cell(colspan: 4)[*CPU backend*],
  [`cpu_panel`], [uint64], [`32`], [Pivots per panel before an update.],
  [`cpu_block_rows`], [uint64], [`96`], [Rows per block in the update.],
  [`cpu_blas_min_rows`], [uint64], [`64`], [Smallest block handed to BLAS.],
  table.cell(colspan: 4)[*GPU scheduling*],
  [`schur_batch`], [uint32], [`0`], [Panels completed between update rounds.],
  [`panel_threads`], [uint32], [`512`], [Threads per group in the panel factorization.],
  [`solve_threads`], [uint32], [`0`], [Threads per front in the solves.],
  table.cell(colspan: 4)[*GPU kernel geometry*],
  [`panel`], [uint32], [`16`], [Pivots per panel.],
  [`schur_block`], [uint32], [`4`], [Tile size in the main update.],
  [`strip_block`], [uint32], [`4`], [Tile height in the strip update.],
  [`narrow_cols`], [uint32], [`32`], [Width where a front takes the wide kernels.],
  [`pivot_threads`], [uint32], [`128`], [Threads per pivot search.],
  [`schur_chunk`], [uint32], [`16`], [Columns staged per update pass.],
)

The sections below explain each group, with the allowed values.

== Solve settings (Python only)

These control refinement and input checking, which live in the Python layer.
Rust has no equivalent yet; @precision shows how to refine from Rust yourself.

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`refinement_steps`], [`-1`, `0`, or a positive integer], [`-1` refines until `refinement_rtol` is reached or refinement stops improving, with a safety cap of 50. `0` switches refinement off. A positive number is a hard cap.],
  [`refinement_rtol`], [between 0 and 1], [The relative residual $norm(b - A x) slash norm(b)$ at which refinement stops.],
  [`refinement_minimize`], [bool], [Scales each correction so it shrinks the residual as much as possible. Costs one extra matrix-vector product per step on the CPU, and roughly halves the steps on badly conditioned matrices.],
  [`equilibrate`], [bool], [Scales rows and columns to similar sizes before the values are rounded to single precision. Can help accuracy when entries differ by many orders of magnitude; costs extra memory and CPU work.],
  [`safe_mode`], [bool], [Re-checks every new matrix passed to `factorize`. Off, a new matrix stored exactly like the analyzed one is read straight into the analyzed layout, which is much faster, and its values are not checked for infinities or NaNs. Either way a different pattern is detected and analyzed again.],
  [`verbose`], [bool], [Prints a short summary per solve: method, ordering, fill-in, memory, timings and residuals. Adds the cost of measuring the residuals.],
)

== Ordering settings

The ordering decides the fill-in, and so most of the memory and time. SCALPEL
is SPARTA's own nested dissection: it runs in parallel and can be tuned. A
slower, better ordering
pays off when one pattern is factored many times, as in a frequency sweep.
Measure the whole workload, because a faster ordering does not always give a
faster factorization.

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`imbalance`], [0.5 to 0.9], [The largest share of the matrix allowed in one half of a split. Near 0.5 asks for even halves; larger values give more freedom to find a small separator, which can reduce fill-in.],
  [`wide_cycles`], [0 or more], [Cycles that widen a separator and search for a smaller replacement. `0` orders faster with more fill-in.],
  [`fm_passes`], [0 or more], [Fiduccia–Mattheyses passes, a refinement that may accept temporary worsening to find a better separator. `0` is much faster and much worse.],
  [`separator_trials`], [1 or more], [Starting separators tried per split; the best is kept. More trials help on irregular meshes.],
)

`RAYON_NUM_THREADS` limits how many CPU threads SCALPEL and the rest of the
analysis use.

*More SCALPEL options, Rust only.* `config.scalpel` has further fields for
ordering experiments. Leave them at their defaults unless you are researching
orderings.

#table(columns: (auto, auto, auto, 1fr),
  head([Field], [Type], [Default], [What it does]),
  [`leaf_size`], [`usize`], [`128`], [Parts at most this size (1 to 256) are ordered by exact minimum degree instead of being cut further.],
  [`parallel_threshold`], [`usize`], [`4096`], [Parts at least this size may be processed in parallel.],
  [`refine_all_candidates`], [`bool`], [`false`], [Refine every candidate separator before choosing one.],
  [`half_levels`], [`bool`], [`false`], [Build starting separators from distances to two ends of the graph.],
  [`balance_quality`], [`bool`], [`true`], [Favour balanced halves as well as small separators.],
  [`refinement_passes`], [`usize`], [`4`], [Greedy improvement sweeps per level. `0` skips them.],
  [`one_sided_refinement`], [`bool`], [`false`], [Move vertices only towards the smaller half.],
  [`coarsen_threshold`], [`usize`], [`512`], [Parts above this size are simplified before cutting, then refined. `0` disables; otherwise at least 32.],
  [`compress`], [`bool`], [`true`], [Merge vertices with identical neighbours before ordering.],
  [`cover_refinement`], [`bool`], [`true`], [Improve separators with a vertex-cover search at their boundaries.],
)

== Front settings

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`nemin`], [1 or more], [The fewest columns a supernode must have before it may merge with its neighbour. Larger values give fewer, wider fronts: better GPU efficiency, but more fill-in and memory. Try 32 when many small fronts dominate.],
  [`max_front`], [1 or more, or unlimited], [The largest front allowed, in rows. Analysis fails if a front would be larger, before any GPU memory is claimed. Use it to reject a pathological matrix early; it is not a total memory budget.],
  [`small_front_rows`], [1 or more], [Fronts up to this height use a serial GPU kernel; taller ones use cooperative kernels. The faster choice depends on the front sizes.],
)

== Numerical settings

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`force_cpu`], [bool], [Factor and solve in double precision on the CPU instead of the GPU. Much slower, but accurate to about sixteen digits. See @precision.],
  [`lu_diag_threshold`], [0 to 1], [LU only. At `0`, each pivot is the largest candidate in its column (strict partial pivoting). Above 0, the diagonal entry is kept as pivot if it is at least this fraction of the largest candidate: `0.1` accepts a diagonal one tenth as large. This saves row swaps but accepts weaker pivots, and *changes the results*. Check residuals if you use it.],
)

== CPU backend settings

These only matter with `force_cpu`.

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`cpu_panel`], [1 or more], [Pivots gathered before a trailing update, on the LU path and on fronts LAPACK cannot factor.],
  [`cpu_block_rows`], [1 or more], [Rows per block in the trailing update, and the unit of work when one front is split across threads. Smaller blocks waste less work but add overhead.],
  [`cpu_blas_min_rows`], [1 or more], [The smallest trailing block handed to BLAS. Raise it to avoid BLAS overhead on small blocks.],
)

== GPU settings (advanced)

The defaults are measured on Apple Silicon. Leave these alone unless you are
tuning for different hardware, and measure every change.

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`schur_batch`], [0 or more], [How many pivot panels complete between update rounds for the rest of the front. `0` runs one round per level of the tree.],
  [`panel_threads`], [`0`, or 32 to 1024], [Threads per group in the panel factorization. `0` means the built-in 512.],
  [`solve_threads`], [`0`, or 32 to 1024], [Threads per front in the solves. `0` chooses per front. Values up to 256 select the 256-thread path, larger values the 512-thread path.],
  [`panel`], [16], [Pivots per panel. Other widths need kernel changes.],
  [`schur_block`], [1 to 8], [Tile size in the main update: each group computes a `16 × schur_block` square tile.],
  [`strip_block`], [1 to 8], [Tile height in the strip update: `16 × strip_block` rows, which must be a multiple of `panel`.],
  [`narrow_cols`], [1 or more], [Fronts wider than this take the wide-front kernels.],
  [`pivot_threads`], [multiple of 32, up to 1024], [Threads cooperating on one pivot search.],
  [`schur_chunk`], [1 or more], [Columns staged together per update pass.],
)

The last six settings are compiled into the GPU kernels. A combination the
process has not used before costs about 300 ms of compilation the first time
a solver is created. Both update kernels must also fit the GPU's 32 KiB of
shared memory per group:

- main update: $2 times (16 times "schur_block") times "schur_chunk" times 8$ bytes,
- strip update: $(16 times "strip_block" + 16) times "schur_chunk" times 8$ bytes.

A combination that breaks a limit is rejected with a message naming the
setting.

== Environment variables

#table(columns: (auto, 1fr),
  head([Variable], [Effect]),
  [`SPARTA_SCHUR_BATCH`], [Overrides the default of `schur_batch`.],
  [`SPARTA_PANEL_THREADS`], [Overrides the default of `panel_threads`.],
  [`SPARTA_SOLVE_THREADS`], [Overrides the default of `solve_threads`.],
  [`SPARTA_LU_DIAG_THRESHOLD`], [Overrides the default of `lu_diag_threshold`.],
  [`RAYON_NUM_THREADS`], [Limits the CPU threads used by the analysis. Set it before the program starts.],
  [`SPARTA_PROFILE`], [`1` prints the time of each stage of analysis and factorization; `2` adds GPU dispatch detail.],
)

Settings you assign explicitly in code win over these variables.

When comparing settings, measure analysis, factorization and solve time, and
the factor memory, over your whole workload. In Rust, `Analysis` reports
`max_front_size()` and `factor_bytes()`, and `Solver` reports
`factor_seconds()`. In Python, `solver.info` and `solver.diagnostics()` give
the same.

= Precision <precision>

This chapter explains why a solve is never exact, how to tell how good an
answer is, and exactly what to do when it is not good enough.

== Why answers are not exact

*Rounding.* The GPU computes in single precision, which keeps about seven
significant digits. Every operation rounds, and a factorization performs
billions of them. The CPU backend uses double precision, about sixteen digits.

*Conditioning.* Some systems amplify small errors. The *condition number* of
$A$ measures by how much: with a condition number of $10^6$, rounding errors in
the seventh digit can grow into errors in the first. Simulation matrices at
low frequencies, or with very fine meshes, are often badly conditioned.

*Residual and error are different.* The *relative residual*
$norm(b - A x) slash norm(b)$ measures how well $x$ satisfies the equations. It
is cheap to compute. The *error* $norm(x - x_"true") slash norm(x_"true")$ is
what you care about, but it cannot be computed without knowing the true
answer. On a badly conditioned matrix the error can be far larger than the
residual. On one measured test system (condition number about $2.4 times 10^9$),
a residual of $4 times 10^(-7)$ came with an error of $8 times 10^(-5)$, two
hundred times larger.

#note[*Ask for a tighter residual than the accuracy you need.* On that system,
an error of $2 times 10^(-8)$ needed `refinement_rtol = 1e-10`.]

== The options at a glance

#table(columns: (auto, auto, auto, auto),
  head([Approach], [Accuracy], [Cost], [Available in]),
  [GPU, no refinement], [residual about $10^(-5)$], [fastest], [Rust, Python],
  [GPU with refinement], [residual $10^(-6)$ by default, or as tight as you ask], [about 9% more time on large systems at the default], [Python built in; Rust with @rust-refine],
  [CPU backend], [double precision, about 16 digits], [much slower], [Rust, Python],
)

*Iterative refinement* is the usual answer. It keeps the fast single-precision
factors, and repeats three steps: compute the residual $r = b - A x$ in double
precision, solve $A d = r$ with the single-precision factors, and add the
correction $d$ to $x$. Each round gains several digits, until the residual
reaches the target.

== What to do: step by step

+ *Keep your data in double precision.* Pass $A$ and $b$ as float64 or
  complex128. Refinement measures the residual against the values you pass, so
  it cannot recover digits that were lost before SPARTA saw them.
+ *Check the factorization* before trusting a solve (@check-quality).
+ *Use refinement,* and set the residual target tighter than the accuracy you
  need (@more-accuracy).
+ *Read how refinement ended.* If it converged, you are done. If it
  stagnated, continue with step 5.
+ *Help the factorization:* turn on `equilibrate`, use the `careful` preset,
  and make sure `lu_diag_threshold` is `0`.
+ *If that is still not enough, switch to the CPU backend* (@cpu). It is
  slower, but computes in double precision throughout.
+ *Verify the final answer* by computing its residual yourself.

== Checking the factorization <check-quality>

*Python.* Right after `factorize()`, before any solve, `solver.quality`
describes how close the factorization came to breaking down. It costs no extra
GPU work.

#code(```python
solver.factorize()
q = solver.quality
print(q.verdict)       # 'good', 'degraded', 'unusable' or 'unknown'
print(q.usable)        # False only when the verdict is 'unusable'
```)

#table(columns: (auto, 1fr),
  head([Verdict], [Meaning and what to do]),
  [`good`], [The factors are sound. Solve normally.],
  [`degraded`], [Some pivots are close to single-precision resolution. Solves are still usable, but rely on refinement and check the result.],
  [`unusable`], [The factorization effectively broke down. Refinement will not recover it: use the CPU backend.],
  [`unknown`], [There is not enough information to judge, which happens on the CPU backend before any refined solve. Check the refinement report or the residual instead.],
)

*Rust.* Two methods give the raw signals:

#code(```rust
let (smallest, largest) = solver.pivot_range().unwrap();
let nudged = solver.perturbed_pivots();
```)

- `pivot_range()` returns the smallest and largest pivot size. A ratio
  `smallest / largest` near single-precision resolution (about $10^(-7)$) is a
  warning; far below it, expect the solve to fail.
- `perturbed_pivots()` counts pivots SPARTA had to nudge away from zero to
  avoid a breakdown. Nonzero means the factors belong to a slightly different
  matrix: use refinement.

Neither a good verdict nor zero perturbed pivots guarantees an accurate
answer. They catch breakdowns; the residual confirms the result.

== Getting more accuracy from the GPU <more-accuracy>

=== In Python

Refinement is on by default, with a target residual of $10^(-6)$:

#code(```python
# Keep double precision.
A = A.astype(np.complex128)
b = b.astype(np.complex128)

# A tighter target than the default 1e-6.
solver = sparta.analyze(A, refinement_rtol=1e-10)
solver.factorize()
x = solver.solve(b)

# One report per right-hand side.
for report in solver.refinement_reports:
    print(report.status, report.corrections,
          report.relative_residual, report.estimated_relative_error)
```)

A refined solve returns complex128, whatever you passed in. Each report tells
you:

#table(columns: (auto, 1fr),
  head([Field], [Meaning]),
  [`status`], [`CONVERGED`: the target was reached. `STAGNATED`: corrections stopped helping; the best answer found is returned. `ITERATION_LIMIT`: `refinement_steps` ran out. `NONFINITE`: the solve produced infinities or NaNs.],
  [`corrections`], [How many corrections were applied.],
  [`relative_residual`], [The final relative residual.],
  [`estimated_relative_error`], [An estimate of the error in $x$ itself, from the size of the last correction. Usually the number you actually care about. It is generous while refinement converges, and unreliable once it has stagnated.],
  [`history`], [The residual after each step.],
)

Other refinement controls:

#code(```python
# At most five corrections.
sparta.analyze(A, refinement_steps=5)
# Refinement off: the bare single-precision solve.
sparta.analyze(A, refinement_steps=0)
# Rescale rows and columns before rounding to single precision.
sparta.analyze(A, equilibrate=True)
# Change the target for an existing solver.
solver.factorize(refinement_rtol=1e-12)
```)

=== In Rust <rust-refine>

Rust solves return the single-precision answer; refinement is up to you. The
function below implements it. It needs a way to multiply your matrix by a
vector in double precision, passed as `matvec`.

#code(```rust
use sparta::{Complex32, Complex64, Solver};

/// Solves A x = b, then corrects x in double precision until
/// ||b - A x|| / ||b|| <= rtol or max_steps corrections have been made.
fn solve_refined(
    solver: &mut Solver,
    matvec: impl Fn(&[Complex64]) -> Vec<Complex64>,
    b: &[Complex64],
    rtol: f64,
    max_steps: usize,
) -> sparta::Result<Vec<Complex64>> {
    let norm = |v: &[Complex64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let target = rtol * norm(b);
    let mut x = vec![Complex64::new(0.0, 0.0); b.len()];
    let mut r = b.to_vec(); // residual b - A x, in double precision
    for _ in 0..=max_steps {
        let scale = norm(&r);
        if scale <= target {
            break;
        }
        // Scale to unit size before rounding, so small residuals keep their digits.
        let rhs: Vec<Complex32> = r
            .iter()
            .map(|v| Complex32::new((v.re / scale) as f32, (v.im / scale) as f32))
            .collect();
        let d = solver.solve(&rhs)?;
        for (xi, di) in x.iter_mut().zip(&d) {
            *xi += Complex64::new(di.re as f64, di.im as f64) * scale;
        }
        let ax = matvec(&x);
        for ((ri, bi), axi) in r.iter_mut().zip(b).zip(&ax) {
            *ri = bi - axi;
        }
    }
    Ok(x)
}
```)

Call it after `factor`:

#code(```rust
solver.factor(&values)?;
let x = solve_refined(&mut solver, |x| my_matrix.multiply(x), &b, 1e-10, 20)?;
```)

The residual must be computed from your original double-precision matrix,
not from the single-precision values you passed to `factor`: that is where the
extra digits come from. If the residual stops shrinking, stop refining and move
to the CPU backend.

== Solving on the CPU in double precision <cpu>

`force_cpu` performs the factorization and the solves in double precision on
the CPU, with the same analysis. It is much slower than the GPU but keeps
about sixteen digits throughout, which is the answer for systems too badly
conditioned for refinement. SPARTA never switches to the CPU by itself: a GPU
failure is reported as an error.

*Python*

#code(```python
solver = sparta.analyze(A, force_cpu=True, refinement_steps=0)
solver.factorize()
x = solver.solve(b)

# Or switch an existing solver, keeping its analysis, and back again.
solver.factorize(force_cpu=True)
solver.factorize(force_cpu=False)
```)

Refinement also works on the CPU backend, but is rarely needed there.

*Rust.* Set `force_cpu` and use the double-precision methods, which take
`Complex64`:

#code(```rust
let config = Config { force_cpu: true, ..Config::default() };
let mut solver = Solver::new(Analysis::ldlt(n, &col_ptr, &row_idx, &config)?)?;
// values and b are &[Complex64]; solve_many_c128 takes several b.
solver.factor_c128(&values)?;
let x = solver.solve_c128(&b)?;
```)

On Linux and Windows, install a BLAS library first, or the CPU backend runs
much more slowly (@blas).

== When refinement does not converge

#table(columns: (1fr, 1fr),
  head([What you see], [What to do]),
  [`status` is `STAGNATED` with a residual above the target], [Try `equilibrate=True`. Then the `careful` preset. Then `force_cpu=True`.],
  [`status` is `ITERATION_LIMIT`], [Raise `refinement_steps`, or set it to `-1`.],
  [`status` is `NONFINITE`, or `quality.verdict` is `unusable`], [The single-precision factorization broke down. Use `force_cpu=True`.],
  [Residual is small but the answer is still inaccurate], [The matrix is badly conditioned. Tighten `refinement_rtol` by two or three orders of magnitude, and read `estimated_relative_error`.],
  [`perturbed_pivots` is nonzero], [Rely on refinement, and check the residual. Consider the CPU backend if refinement stagnates.],
  [You set `lu_diag_threshold` above 0], [Set it back to `0`: strict pivoting is the most stable.],
)

== Checking the final answer

Whatever route you take, the residual in double precision is the final check:

#code(```python
residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
```)

In Rust, compute $norm(b - A x) slash norm(b)$ with your own double-precision
matrix-vector product. A small residual together with a good `quality`
verdict, or a converged refinement report with a small
`estimated_relative_error`, is as much assurance as a solver can give.
