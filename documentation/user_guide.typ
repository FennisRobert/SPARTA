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
      align(left, text(tracking: 0.08em)[SPARTA · SPARSE DIRECT SOLVER FOR FEM]),
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
    box(baseline: -0.75em, text(tracking: 0.08em)[· SPARSE DIRECT SOLVER FOR FEM])
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
    text(font: serif, style: "italic", size: 14.5pt, fill: rgb("#e6e0e1"))[The sparse direct solver for finite-element
      problems, tuned to keep every core busy, on the GPU or CPU.]
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
      fact[Real and complex][single precision, refined to double],
      fact[Built for FEM][planned once, every core kept busy],
      fact[Own dense kernels][up to 91% of peak, no BLAS to install])
    v(16pt)
    text(size: 9pt, tracking: 0.12em, fill: white)[ROBERT FENNIS #h(0.6em) · #h(0.6em) MILAN ROTHER]
  }))
}))

#outline(title: [Contents], depth: 2)
#pagebreak()

= Introduction

SPARTA solves $A x = b$ for the large sparse matrices that finite-element
(FEM) simulations produce, one per frequency or time step. It is a direct
solver: it factors $A$, then solves.

SPARTA is built for FEM systems only. Many small, measured choices keep every
core busy, so the factorization runs close to the machine's limits on memory
bandwidth and arithmetic (@fast). You do not need to install BLAS, LAPACK or
any other maths library.

This guide covers installing SPARTA, solving, choosing settings and getting
the accuracy you need.

== The three steps

Every solve goes through three steps. You can repeat each step on its own.

+ *Analyze.* SPARTA looks only at _where_ the nonzeros are (the *pattern*),
  chooses the elimination order and plans the memory. This runs on the CPU.
+ *Factor.* SPARTA splits $A$ into *factors* using its values. This is the
  expensive step. It runs on the GPU, unless you select the CPU.
+ *Solve.* SPARTA finds $x$ for a right-hand side $b$ from the factors. This
  is cheap.

What you repeat depends on what changed:

- *A new right-hand side* needs only a new solve.
- *New values in the same pattern*, such as the next frequency of a sweep, need
  a new factorization, but reuse the analysis and the memory it reserved.
- *A new pattern* needs a new analysis. In Python, if you pass `factorize` a
  matrix with a different pattern, it runs the analysis for you.

The factors have nonzeros where $A$ has zeros. This is the *fill-in*, often ten
times as many nonzeros as $A$ itself. The elimination order, the *ordering*,
decides how much fill-in there is, and so most of the memory and time. The work
is done in dense blocks called *fronts*. The columns of one front form a
*supernode*.

== Why SPARTA is fast <fast>

This section shows where SPARTA's speed comes from, so you can judge how it
applies to your own problems.

A direct solver's time is the number of operations divided by the rate at
which they run. For FEM systems, the number of operations is already as low as
it can be. SPARTA is therefore built to raise the rate: to keep every core busy
and close to its peak.

*The number of operations.* SPARTA uses nested dissection: it splits the mesh
in two along a small separator, orders each half the same way, and eliminates
the separator last. On FEM meshes, no other order does better, up to a
constant factor (George, 1973; Hoffman, Martin and Rose, 1973; Lipton, Rose
and Tarjan, 1979):

#table(columns: (auto, 1fr, 1fr),
  head([Mesh, $n$ unknowns], [Operations to factor], [Memory for the factors]),
  [2D], [proportional to $n^(1.5)$], [proportional to $n log n$],
  [3D], [proportional to $n^2$], [proportional to $n^(4\/3)$],
)

*The rate.* Each choice below was measured on FEM test cases and kept because
it made them faster.

- *Plan once, factor many times.* The analysis fixes the ordering, the schedule
  and every block of memory. For a new matrix with the same pattern, SPARTA
  goes straight to the numerical factorization.
- *Wide fronts.* SPARTA merges neighbouring supernodes into wider fronts, even
  when that stores a few extra zeros. Wider fronts run closer to peak.
- *One large update per node.* In CPU mode, each node's update goes straight
  into the columns of the nodes above it.
- *Parallelism that never competes for cores.* Near the bottom of the
  elimination tree, there are many small fronts, and each runs on one core.
  Near the top, there are a few large fronts, and each spreads over every
  core. The dense kernels never add threads that compete with these.
- *Pivots stay in their front.* When a pivot is tiny, SPARTA raises it instead
  of passing it on (`static_pivot`). As a result, the plan never changes during
  factoring.
- *Real arithmetic where the values are real.* In many FEM models, only the
  port and boundary unknowns have complex values. SPARTA factors the parts that
  stay real in real arithmetic, which is about four times cheaper
  (`real_split`).
- *Complex symmetric $L D L^T$.* Lossy materials and absorbing boundaries give
  complex symmetric matrices. SPARTA factors these as $L D L^T$, which costs
  half as much as LU.
- *Solves in blocks.* SPARTA pushes many right-hand sides through the factors
  in one pass. Port projections touch only the parts of the factors they need
  (@ports).
- *No round trips in GPU mode.* The whole factorization runs on the GPU,
  without returning to the CPU between steps.

*The speed of each operation.* Most of the arithmetic happens in large fronts
near the top of the elimination tree, as matrix products. SPARTA runs these on
its own dense kernels, SpartaBLAS, with a separate kernel set for each
processor family (@blas). On an Apple M4 Pro, against the processor's measured
peak:

#table(columns: (1fr, auto, auto, auto),
  head([Product], [Peak], [SpartaBLAS], [Of peak]),
  [`f64`, one thread, matrix unit], [499 GFLOP/s], [450 GFLOP/s], [90%],
  [`complex128`, one thread, matrix unit], [499 GFLOP/s], [453 GFLOP/s], [91%],
  [`f32`, one thread, matrix unit], [1996 GFLOP/s], [1634 GFLOP/s], [82%],
  [`complex64`, one thread, matrix unit], [1996 GFLOP/s], [1763 GFLOP/s], [88%],
  [`f64`, one core, vector unit], [58 GFLOP/s], [52 to 55 GFLOP/s], [90 to 95%],
  [`complex128`, all 14 cores], [1255 GFLOP/s], [902 GFLOP/s], [72%],
)

The products are $512 times 512 times 512$ on one thread and $3000 times 3000
times 400$ on all cores, in the form the factorization uses. The x86 kernel
sets are written but not yet measured on x86 hardware.

*What SPARTA is not designed for.* Two kinds of sparse systems fit SPARTA
poorly:

- *Badly scaled, strongly indefinite systems*, such as some saddle-point
  problems. These need pivots that move between fronts to stay accurate.
  SPARTA keeps pivots in their front, so it can lose accuracy here. When that
  happens, SPARTA reports it (@check-quality).
- *Matrices that do not come from a mesh*, such as circuit or network
  matrices. These have few large fronts, so SPARTA's dense kernels hardly help.

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

Each elimination step divides by one entry, the *pivot*. To avoid pivots near
zero, both methods swap rows or use $2 times 2$ blocks.

== Single and double precision

The GPU computes in single precision, which holds about seven digits. A
typical FEM matrix solves to a *relative residual* $norm(b - A x) slash norm(b)$
of about $10^(-5)$. For more accuracy, refine the answer in double precision,
or factor in double precision on the CPU (@precision).

The arithmetic follows the values you pass. Real values are factored as float32
on the GPU or float64 on the CPU. Complex values are factored as complex64 or
complex128.

== What is supported

#table(columns: (auto, 1fr),
  head([], [Support]),
  [GPU], [Apple Silicon (M1 or later) through Metal. AMD and NVIDIA GPUs on
    Windows and Linux through Vulkan.],
  [CPU], [A double-precision backend on any 64-bit machine, used when you ask
    for it, on SPARTA's own dense kernels (@blas).],
  [Languages], [Rust and Python, covered in this guide, and MATLAB and Julia
    (@bindings). A C and C++ interface exists as well; see
    `crates/sparta-capi/README.md`.],
  [Matrices], [Square, real or complex, stored in compressed sparse column
    form (Rust) or any SciPy sparse format (Python). Real matrices are
    factored with real arithmetic.],
  [Ordering], [SCALPEL, SPARTA's own parallel nested dissection.],
)

= Installation

== Requirements per operating system

*macOS.* An Apple Silicon Mac (M1 or later) running macOS 13 or later runs
everything, including the GPU. An Intel Mac runs the analysis and the
double-precision CPU backend, but has no GPU.

*Windows.* Any 64-bit machine runs the analysis and the CPU backend. AMD and
NVIDIA GPUs run through the Vulkan backend, which was tested on an NVIDIA
RTX 3080 and an AMD Radeon integrated GPU. A normal install includes it. To run it, you need a GPU
driver with Vulkan 1.3. Without one, `gpu_available()` is false, and
`force_cpu=True` selects the CPU backend.

*Linux.* The same as Windows: the analysis, the CPU backend and the Vulkan
backend, tested on two NVIDIA RTX A5000s. The NVIDIA driver on a server or
headless install often comes without its Vulkan part, so the GPU works for
CUDA but `gpu_available()` is false. `sparta.gpu_status()` then says which part
is missing and, when an NVIDIA driver is loaded, which package to install. To
add it on Ubuntu or Debian, install the Vulkan loader and the Vulkan driver that
match the installed driver version (580 here, or `libnvidia-gl-580-server` for
the server driver):

```sh
sudo apt install libvulkan1 vulkan-tools libnvidia-gl-580
vulkaninfo --summary      # lists the GPUs Vulkan can use
```

On Fedora and RHEL, the loader is `vulkan-loader`, and the NVIDIA driver
package includes its Vulkan part. In a Docker container,
`NVIDIA_DRIVER_CAPABILITIES` must include `graphics`.

== Packages you need <requirements>

#table(columns: (auto, 1fr),
  head([Package], [What it is for]),
  [Rust toolchain (`rustc`, `cargo`)], [Compiling SPARTA. Needed for every
    install route except a prebuilt Python wheel.],
  [Xcode command line tools], [macOS only: compiling the Metal bridge.],
  [git], [Downloading SPARTA from GitHub.],
  [Python 3.9 or later, with pip], [The Python package.],
  [NumPy and SciPy], [Used by the Python package. pip installs them automatically.],
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

For Python, install SPARTA from PyPI (see @install-python). That needs no
Rust and no compiler on the platforms listed there. Every other route compiles
SPARTA from source.

SPARTA is not published on crates.io. For Rust, pick the route that matches
what you have:

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

=== Python <install-python>

Install SPARTA into the Python environment you run your simulations in. The
package is imported as `sparta`.

*1. From PyPI.*

#code(```sh
pip install sparta-solver
```)

PyPI has prebuilt wheels for Python 3.9 and later on these platforms, so they
need no Rust and no compiler:

- macOS on Apple Silicon: Metal GPU and CPU.
- macOS on Intel: CPU.
- Linux on x86_64 and aarch64: Vulkan GPU and CPU.
- Windows on x86_64: Vulkan GPU and CPU.

On any other platform, pip downloads the source and compiles it. That needs
the Rust toolchain from @requirements.

*2. From GitHub.* This installs the latest, unreleased code and needs the
Rust toolchain:

#code(```sh
pip install "git+https://github.com/FennisRobert/SPARTA.git"
```)

*3. From a copy of the source.* This also needs the Rust toolchain. Unpack
the source, then install from its top folder, the one containing
`pyproject.toml`:

#code(```sh
cd path/to/SPARTA
pip install .
```)

== Checking the installation

*Rust:* build the example in @quick-rust with `cargo run --release`. It prints
the solution of a small system.

*Python:*

#code(```python
import sparta
print(sparta.__version__)
print(sparta.gpu_available())   # True on an Apple Silicon Mac
print(sparta.gpu_status())      # the GPU in use, or why there is none
```)

If `gpu_available()` prints `False`, `gpu_status()` gives the reason. On an
Apple Silicon Mac, the process cannot see the GPU. This happens in some
sandboxed or remote sessions. Run from a normal Terminal window instead. On
Windows and Linux, the reason names the missing Vulkan loader or driver, and
on Linux with an NVIDIA driver loaded, the package to install.

== The CPU dense kernels <blas>

In CPU mode, SPARTA runs its dense work on SpartaBLAS, its own kernels. The
dense work is the matrix products, the triangular solves and the factorization
of each front's diagonal block. You do not need to install or link anything,
on any platform.

On Apple M1 to M3 and on Intel Macs, SPARTA uses Apple's Accelerate instead,
which comes with macOS. These Macs have a matrix unit that only Apple's
libraries can program, so Accelerate is faster there than SpartaBLAS's `neon`
set. From M4 on, the same unit is programmable through SME, and the `sme512`
set is as fast as Accelerate on our FEM test cases.

SpartaBLAS has one kernel set per processor family. On first use, it picks the
best set the processor supports:

#table(columns: (auto, 1fr, auto),
  head([Set], [Processors], [Status]),
  [`sme512`], [Arm SME2 matrix unit at a 512-bit vector length: Apple M4 and later], [Measured],
  [`neon`], [Every 64-bit Arm processor], [Measured],
  [`avx512`], [x86-64 with AVX-512: Intel Xeon since Skylake-SP, AMD Zen 4 and later], [Not yet run on x86],
  [`avx2`], [x86-64 with AVX2 and FMA: Intel since Haswell, AMD since Zen 1], [Not yet run on x86],
  [`scalar`], [Anything else], [Fallback],
)

To see which set is in use:

#code(```python
import sparta
print(sparta.blas_library())   # for example "SpartaBLAS (sme512)" or "Accelerate"
```)

In Rust, `sparta::blas_library()` returns the same text. To force a set for
testing, set `SPARTA_KERNEL` before the first CPU factorization. On a Mac,
`SPARTA_KERNEL=accelerate` forces Accelerate. If the
processor cannot run that set, SPARTA prints a message and uses the best
available set instead.

#code(```sh
SPARTA_KERNEL=neon python my_simulation.py
```)

SpartaBLAS starts no threads of its own. Large calls split across the same
thread pool as the rest of SPARTA, so `threads` limits them too. Accelerate
uses its own threads, but only for a front that has the whole machine to
itself. On macOS 15 and later, setting `threads` keeps Accelerate on one
thread per call.

= Quick use

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
use sparta::{Analysis, Config, Solver};

fn main() -> sparta::Result<()> {
    // A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]], symmetric: lower triangle only.
    let col_ptr = [0, 2, 4, 5];
    let row_idx = [0, 1, 1, 2, 2];
    let values: [f32; 5] = [4.0, 1.0, 3.0, 1.0, 2.0];
    let b: [f32; 3] = [1.0, 2.0, 3.0];

    // 1. Analyze the pattern.
    let analysis = Analysis::ldlt(3, &col_ptr, &row_idx, &Config::default())?;
    let mut solver = Solver::new(analysis)?;
    // 2. Factor the values.
    solver.factor_values(&values)?;
    // 3. Solve for x: one right-hand side.
    let x = solver.solve_values(&b, 1)?;
    println!("{x:?}");
    // 4. Free the GPU memory.
    solver.destroy();
    Ok(())
}
```)

`Analysis::ldlt` is for symmetric matrices. For any other matrix, use
`Analysis::lu` and pass the whole matrix.

`factor_values` and `solve_values` take `f32`, `f64`, `Complex32` or
`Complex64`, and the type decides the arithmetic. The example above is a real
matrix, so it passes `f32`. For a complex matrix, pass `Complex32` values and
a `Complex32` right-hand side instead:

#code(```rust
use sparta::Complex32;
let values = [4.0, 1.0, 3.0, 1.0, 2.0].map(|v| Complex32::new(v, 0.5));
let b = [1.0, 2.0, 3.0].map(|v| Complex32::new(v, 0.0));
solver.factor_values(&values)?;
let x = solver.solve_values(&b, 1)?;   // x is Complex32
```)

- *Which types where.* The GPU takes `f32` and `Complex32`. The CPU backend
  (`force_cpu`, @cpu) takes all four and computes in the precision you pass.
- *Speed.* In CPU mode, a real matrix factors about twice as fast as the same
  matrix stored as complex. In GPU mode, real matrices still run through the
  complex kernels, so they are not faster yet.
- *Right-hand sides.* A right-hand side must have the precision of the
  factors. You can solve a complex right-hand side against real factors.
  SPARTA refuses a real right-hand side against complex factors, because its
  solution is complex.

`factor` and `solve` are shorthands for `Complex32` values and one
right-hand side.

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
symmetric, LU otherwise. The arithmetic follows the dtype. This $A$ is real, so
SPARTA factors it in real arithmetic, and `x` is a real array. A complex $A$ or
a complex `b` gives a complex `x`:

#code(```python
A = csc_matrix([[4.0 + 0.5j, 1.0, 0.0],
                [1.0, 3.0 + 0.5j, 1.0],
                [0.0, 1.0, 2.0 + 0.5j]])   # complex: complex arithmetic
x = sparta.analyze(A).factorize().solve(b)  # x is complex128
```)

`solver.compute_dtype` reports the arithmetic: float32 or complex64 on the
GPU. By default, Python also refines the answer in double precision until the
relative residual is below $10^(-6)$ (@precision).

== Solving again

The three steps can be repeated separately.

*A new right-hand side:* Call `solve` again. Several right-hand sides are
faster in one call:

#code(```rust
// Rust: the columns end to end; column j is b[j*n..(j+1)*n].
let x = solver.solve_values(&b, nrhs)?;
```)
#code(```python
# Python: B has shape (n, nrhs), one right-hand side per column.
X = solver.solve_many(B)
```)

*New values with the same pattern*, such as the next frequency: Factor again.
SPARTA reuses the analysis and the memory.

#code(```rust
// Rust: the values in the same CSC order as the first time. They may change
// type, say from real to complex; the arithmetic follows.
solver.factor_values(&new_values)?;
```)
#code(```python
# Python: same pattern, new values.
solver.factorize(matrix=A_next)
```)

*Only a few numbers from the solution*, such as a port matrix: see
@ports.

*A new pattern:* Start again from the analysis. In Python, `factorize` does
this for you. If you pass a matrix with a different pattern or size, it
analyzes again with the same settings, then factors.

`solver.reanalyses` counts these extra analyses. Each one costs a full analysis
and new factor memory. If the count keeps rising in a loop, your pattern
changes between calls, for example because explicit zeros come and go.

== Port matrices and sparse right-hand sides <ports>

In a frequency sweep, you often need only a few numbers from each solution:
the port matrix of the model, or the field at a few chosen unknowns. SPARTA
can compute those directly, without forming the full solution.

- *B, what you excite:* an $n times p$ sparse matrix with one column per
  excitation, such as one port. Each column is nonzero only on that port's
  unknowns.
- *W, what you measure:* an $n times q$ sparse matrix with one column per
  quantity, each a weighted sum of unknowns.

The result is $W^T A^(-1) B$, a small $q times p$ matrix. Entry $(i, j)$ is
measurement $i$ when column $j$ of B is excited. The transpose is plain, never
conjugated. With $W = B$, the result is the port matrix $B^T A^(-1) B$. With
one column per chosen unknown, it gives the solution's values there.

#code(```python
X = solver.solve_many(B)              # B may be SciPy sparse: the full solution, faster
Z = solver.solve_many(B, project=B)   # the p x p port matrix
V = solver.solve_many(B, project=W)   # W^T A^-1 B, q x p
v = solver.solve(b, project=W)        # one right-hand side: W^T x
```)
#code(```rust
use sparta::SparseColumns;
// CSC arrays, numbered as the matrix: column j has rows[ptr[j]..ptr[j+1]].
let b = SparseColumns::new(n, &ptr, &rows, &values)?;
let x = solver.solve_sparse(&b)?;              // n values per column, end to end
let z = solver.solve_projected(&b, None)?;     // B^T A^-1 B, p x p, column-major
let v = solver.solve_projected(&b, Some(&w))?; // W^T A^-1 B, q x p, column-major
```)

*How it runs.* For $L D L^T$ factors in CPU mode, SPARTA takes a fast route.
A column of B only reaches the fronts on the path from its unknowns to the top
of the elimination tree, so a projection only touches those fronts. It forms no
full solution and builds no $n times p$ block.

For GPU factors, the hybrid factorization and LU, SPARTA solves as usual and
then projects. The answer is the same, but not faster. In Python,
`solver.diagnostics()['solve_route']` says which route the latest solve took.

*Accuracy.* In Python, SPARTA takes the fast route only when no pivot was
perturbed. A sparse solve without `project` checks its residual and, if
needed, refines on the usual route. A fast projection forms no full solution,
so it has no refinement report. For a symmetric matrix, `project=B` returns an
exactly symmetric result on every route.

*What it saves*, on the CPU (M4 Pro, complex128) with ports of six unknowns:

#table(
  columns: 5,
  [*System*], [*Ports*], [*Solve, then project*], [*Projected directly*], [*With the factorization*],
  [109k unknowns], [4], [0.030 s], [0.010 s], [1.06× faster],
  [109k unknowns], [100], [0.363 s], [0.028 s], [2.0× faster],
  [887k unknowns], [20], [0.782 s], [0.126 s], [1.13× faster],
  [887k unknowns], [100], [7.64 s], [0.20 s], [2.5× faster],
)

The more ports you have, the more it saves. With only a few ports, the
factorization takes most of the time either way.

*If you know separate port-matrix calls.* Some solvers offer
`port_matrix(B)` and `solve_sparse(B)` as separate calls. In SPARTA:
- `port_matrix(B)` is `solve_many(B, project=B)`, or
  `solve_projected(&b, None)` in Rust.
- `solve_sparse(B)` is `solve_many(B)` with a sparse B, or `solve_sparse(&b)`
  in Rust.
- You do not need to ask how far a column reaches into the factors. SPARTA
  picks the route itself.

SPARTA's versions work on every factorization and also take a separate W. C,
Julia and MATLAB do not offer them yet.

== Freeing memory

A factorization of a large system can hold more than a gigabyte of GPU
memory until its solver is freed. Call `destroy()` when you are done, as in
the examples above. In Rust, dropping the solver also frees it. In Python,
without `destroy()`, the memory stays claimed as long as anything refers to
the solver. So use `destroy()` or a `with` block:

#code(```python
with sparta.analyze(A) as solver:
    solver.factorize()
    x = solver.solve(b)
```)

= MATLAB and Julia <bindings>

Both bindings call SPARTA's C library, so they share its limits:

- They solve on the GPU only. The CPU backend (`force_cpu`) and the tuning
  search (@optimize) are not available.
- The binding itself refines in double precision, more simply than Python
  does. It stops when the residual is small enough or stops improving.

Build the C library from the repository's top folder first. This needs only
Rust:

#code(```sh
cargo build --release -p sparta-capi
```)

Settings have the names and meanings of @all-settings, plus:

- `kind` chooses between LDLᵀ and LU, as `mtype` does in Python.
- `preset` picks the starting point (@presets).
- `refinement_steps` and `refinement_rtol` control refinement, as they do in
  Python.

== MATLAB

#block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)[
  *Not yet tested in MATLAB.*
  - The binding was written without access to MATLAB.
  - Its C part has been run against the library through a stand-in for
    MATLAB, but the MATLAB class itself has not been run.
  - Run `test_sparta` first, and report a failure as a GitHub issue.]

*Requirements:*

- MATLAB R2018a or later.
- A C compiler that `mex -setup C` accepts. On Windows that is Visual Studio.
- On Apple Silicon, the native MATLAB, R2023b or later.

*Building:* in MATLAB, from `bindings/matlab`:

#code(```matlab
build_sparta_mex     % compiles the gateway against target/release
test_sparta          % checks the binding end to end
```)

Then add that folder to the MATLAB path.

*Solving:* `SpartaSolver` works like MATLAB's own `decomposition`:

#code(```matlab
S = SpartaSolver(A);                    % analyze and factor
x = S \ b;                              % solve; b may have several columns

S = SpartaSolver(A, 'preset', 'sweep', 'nemin', 32);
for f = frequencies
    factorize(S, K - f^2 * M);          % same pattern: plan and memory reused
    x = S \ b;
end
clear S                                 % frees the GPU memory
```)

Settings go in as name-value pairs. Errors carry identifiers such as
`sparta:pivot` for a singular matrix. `help SpartaSolver` and
`bindings/matlab/README.md` list the rest.

== Julia

*Requirements:* Julia 1.9 or later. The package calls the library directly,
so it needs no compiler.

*Installing:* add the package to an environment, then run its tests:

#code(```julia
using Pkg
Pkg.develop(path = "bindings/julia")
Pkg.test("SPARTA")
```)

It loads the library from `target/release`. To use a library elsewhere, set
the environment variable `SPARTA_LIB` to its path.

*Solving:*

#code(```julia
using SPARTA, SparseArrays

F = SPARTA.Solver(A)                    # analyze and factor
x = F \ b                               # solve; b may have several columns

F = SPARTA.Solver(A; preset = :sweep, nemin = 32)
for f in frequencies
    factorize!(F, K - f^2 * M)          # same pattern: plan and memory reused
    x = F \ b
end
close(F)                                # frees the GPU memory now
```)

Settings are keyword arguments. A failed call throws `SpartaError`, whose
`kind` is one of the following:

- `:pivot`: the matrix is singular or too badly scaled.
- `:unsupported`: no GPU is available.
- `:device`: the GPU failed.
- `:invalid`: an input or setting was rejected.

Several tasks may share one solver, because SPARTA runs its calls one at a
time.
`bindings/julia/README.md` lists the rest.

== Both bindings

- *Pass the whole matrix*, both triangles, even when it is symmetric.
  Refinement multiplies by it.
- *Changing pattern.* A later matrix may lack some entries of the first, for
  example because MATLAB drops entries that compute to exactly zero. In that
  case, SPARTA keeps the plan and puts zeros in their place. If a later matrix
  has an entry the plan does not have, SPARTA analyzes again.
- *Precision.* Answers come back in double precision. With `refinement_steps`
  set to 0, they have the GPU's single-precision accuracy.
- *Real or complex.* Julia follows the element type. A real matrix is factored
  in real arithmetic and, with a real `b`, returns `Float64`. A complex matrix
  returns `ComplexF64`. MATLAB still factors every matrix as complex and
  returns `complex double`.

= Settings and configuration

The defaults are measured and suit most systems. Change a setting only when a
measurement shows it helps (@optimize).

== How settings work

Settings belong to one of two moments:

- *Analysis settings* shape the plan: the ordering, the size of fronts and
  the GPU kernel shapes. They are fixed once the matrix is analyzed. To change
  one, analyze again.
- *Solve settings* (Python only) control refinement and checking. You can
  change them at every factorization.

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
entries: `sparta.EPS`, which starts at `1e-10`. Set it before you call
`analyze`. `sparta.EPS = 0` demands exact symmetry. With a larger tolerance,
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

*Unknown keywords.* `analyze`, `factorize` and `Config.with_` ignore a
keyword they do not know and issue a `sparta.UnusedSettingWarning`. So
settings written for another SPARTA version never stop a run. `factorize`
silently ignores settings that only `analyze` can apply, such as `nemin`. So
you can pass one settings dict to both calls.

*Which value wins.* A keyword argument overrides the same setting in
`config`, with one exception. `max_front` has a default of its own, `None`, so
passing it does not override the config. To remove a config's front limit,
change the config itself:

#code(```python
config = config.with_(max_front=None)
```)

In a `Config`, `ordering` is stored as text (`"scalpel"`). So use the enum's
`.value` in a `Config`, and the enum itself as a keyword argument.

*Changing settings later.* To change an existing solver, pass the solve
settings, `force_cpu` or `hybrid` to `factorize`. Settings you leave out keep
their current value. All other settings need a new `analyze`.

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

The enums also contain members reserved for later versions. If you select
one, SPARTA raises `NotImplementedError` instead of silently doing something
else.

== Presets <presets>

A preset is a named starting point for a kind of workload. The best settings
still depend on your matrices, so measure before you keep one.

#table(columns: (auto, auto, 1fr),
  head([Rust], [Python], [Use it for, and what it changes]),
  [`Balanced`], [`"balanced"`], [The defaults. One factorization of an unstructured mesh.],
  [`Sweep`], [`"sweep"`], [One pattern factored many times. `wide_cycles = 6` and `separator_trials = 8`: more ordering work, paid once.],
  [`OneShot`], [`"one-shot"`], [Factor once and discard. `wide_cycles = 0`, `separator_trials = 1`, and no spectral or all-candidate refinement: a fast ordering with more fill-in.],
  [`StructuredGrid`], [`"structured-grid"`], [Regular grids. `nemin = 32`: wider fronts.],
  [`Careful`], [`"careful"`], [Ill-conditioned systems. `imbalance = 0.6`, a more balanced split, and strict LU pivoting (`lu_diag_threshold = 0`) even when `SPARTA_LU_DIAG_THRESHOLD` is set.],
)

To judge a change, measure analysis, factorization and solve time and the
factor memory over your whole workload. In Rust, `Analysis` reports
`max_front_size()` and `factor_bytes()`, and `Solver` reports
`factor_seconds()`. In Python, `solver.info` and `solver.diagnostics()` give
the same.

== Tuning to your computer <optimize>

`sparta.optimize` factors your matrices with many candidate settings. It
keeps what is clearly faster and no less accurate, and saves the result under
a name. Use it instead of guessing at settings.

#code(```python
result = sparta.optimize([A1, (A2, b2)], name="antenna")
solver = sparta.analyze(A, config="antenna")      # any later session
```)

Pass matrices like the ones you will factor, at representative sizes. The
best settings depend on the matrices as much as on the hardware. A problem is
a sparse matrix, or a `(matrix, rhs)` pair with a vector or an `(n, k)` block.
Without problems, `sparta.optimize()` tunes on two standard Helmholtz systems
of 14k and 47k unknowns. This gives a general tuning for the computer.

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

*How it searches.* It changes one setting at a time. First it tunes the
ordering and front settings, on whichever backend is faster. Then it tunes
each backend's own settings. It times each candidate as the median of several
factorize-and-solve runs. It never changes `lu_diag_threshold`, because that
setting trades accuracy for speed. On an M4 Pro, a full run on the standard
problems takes one to two minutes.

*What counts as faster.* A change is kept only when all of these hold:

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

Settings affect each other: a larger `nemin` changes which fronts exist, and
so which kernel geometry suits them. With `iterations=3` the search makes up
to three passes, each starting from what the last one found.

#code(```python
result = sparta.optimize([A1, A2], name="antenna", iterations=3)
```)

A pass that changes nothing ends the search early, so a large value costs
nothing once the settings have settled:

#code(```text
[ 66/286 0:15 | ~0:48 left] iteration 2 of 3: starting again from the settings found so far
[126/130 0:28 | ~0:01 left] iteration 2 changed nothing: the settings have settled, skipping the rest
...
  Iterations: 2 of 3, stopped early.
```)

Each pass takes about as long as the first. `result.iterations` gives
`(run, asked)`.

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

== All settings <all-settings>

Types are given in a language-neutral form. `uint32` is `u32` in Rust and
`int` in Python; `uint64` is `usize` and `int`; `float64` is `f64` and
`float`; "optional" is `Option` in Rust and `None` in Python.

#table(columns: (auto, auto, auto, 1fr),
  head([Name], [Type], [Default], [What it is for]),
  table.cell(colspan: 4)[*Solve settings* (Python only)],
  [`refinement_steps`], [int], [`-1`], [Limits the refinement corrections per right-hand side. `-1` takes as many as needed.],
  [`refinement_rtol`], [float64], [`1e-6`], [Sets the relative residual at which refinement stops.],
  [`refinement_minimize`], [bool], [`True`], [Makes each correction shrink the residual as much as possible, so fewer steps are needed.],
  [`equilibrate`], [bool], [`False`], [Balances row and column sizes before the conversion to single precision. This can improve accuracy.],
  [`safe_mode`], [bool], [`False`], [Checks every new matrix passed to `factorize`. This is slower, but catches infinities and NaNs.],
  [`verbose`], [bool], [`False`], [Prints a short summary after each solve.],
  [`mtype`, `path`, `triangle`], [enum, enum, text], [automatic, `"full"`], [These tell SPARTA how to read and factor the matrix.],
  table.cell(colspan: 4)[*Ordering*],
  [`ordering`], [enum], [SCALPEL], [Selects the nested-dissection implementation. SCALPEL is the only one.],
  [`imbalance`], [float64], [`0.8`], [Only used by SCALPEL. Sets how unequal the two halves of a split may be.],
  [`wide_cycles`], [uint64], [`4`], [Only used by SCALPEL. More cycles give less fill, but the ordering takes longer.],
  [`fm_passes`], [uint64], [`4`], [Only used by SCALPEL. More passes give less fill, but the ordering takes longer.],
  [`separator_trials`], [uint64], [`4`], [Only used by SCALPEL. More trials can find a better split, but the ordering takes longer.],
  table.cell(colspan: 4)[*Fronts*],
  [`nemin`], [uint64], [`16`], [Controls how eagerly columns merge into wider blocks. Larger values suit the GPU, but store more fill.],
  [`max_front`], [uint64, optional], [unlimited], [Makes the analysis fail when a dense block would have more rows than this.],
  [`small_front_rows`], [uint32], [`32`], [Sets the front height where the GPU switches from its serial kernel to cooperative kernels.],
  table.cell(colspan: 4)[*Numerics*],
  [`force_cpu`], [bool], [`False`], [Factors and solves on the CPU in double precision instead of on the GPU. Only available in Python and Rust.],
  [`hybrid_cpu`], [float64], [`0.0`], [Lets the CPU do this share of the factorization while the GPU does the rest. Only used on Metal. In Python, `hybrid=True` turns it on.],
  [`lu_diag_threshold`], [float64], [`0.0`], [Only affects LU factorization. Above 0, SPARTA prefers the diagonal as pivot, which is faster but less stable.],
  [`supernodal`], [bool], [`True`], [Speeds up LDLᵀ factorization and lowers its memory. Expected to be strictly better than the multifrontal factorization, which is used when it is turned off, with the same accuracy. Turn it off to check whether your matrix is an exception (@numerics).],
  [`real_split`], [bool], [`True`], [Speeds up complex LDLᵀ factorization when parts of the matrix are real. Turn it off only if the GPU's first factorizations must all take the same time.],
  [`static_pivot`], [float64], [`1e-12`], [Only affects LDLᵀ factorization. Raises tiny pivots to this size, relative to the largest entry, so SPARTA never divides by a near-zero. `0` replaces exact zeros only.],
  table.cell(colspan: 4)[*CPU backend*],
  [`cpu_panel`], [uint64], [`32`], [Only used in CPU mode. Sets how many pivots are gathered before an update.],
  [`cpu_block_rows`], [uint64], [`96`], [Only used in CPU mode. Sets the rows per block of the trailing update.],
  [`cpu_blas_min_rows`], [uint64], [`64`], [Only used in CPU mode. Blocks smaller than this skip the dense kernels.],
  table.cell(colspan: 4)[*Threads*],
  [`threads`], [uint64], [`0`], [Sets the CPU threads for SPARTA's own work. With `0`, SPARTA shares the process-wide pool.],
  table.cell(colspan: 4)[*GPU scheduling*],
  [`schur_batch`], [uint32], [`0`], [Sets how many panels the GPU completes between two update drains.],
  [`panel_threads`], [uint32], [`512`], [Sets the threads per threadgroup in the GPU panel factorization.],
  [`solve_threads`], [uint32], [`0`], [Sets the threads per block in the GPU solves.],
  table.cell(colspan: 4)[*GPU kernel geometry*],
  [`panel`], [uint32], [`16`], [Sets the pivots per GPU panel.],
  [`schur_block`], [uint32], [`4`], [Sets the register block of the GPU Schur update.],
  [`strip_block`], [uint32], [`4`], [Sets the register block of the GPU strip update.],
  [`narrow_cols`], [uint32], [`32`], [Sets the width above which a block uses the GPU's wide kernels.],
  [`pivot_threads`], [uint32], [`128`], [Sets the GPU threads that search for one pivot together.],
  [`schur_chunk`], [uint32], [`16`], [Sets the panel columns staged per GPU Schur pass.],
)

The sections below explain each group, with the allowed values.

== Solve settings (Python only)

These settings control refinement and input checking, which live in the
Python layer. Rust does not have them yet. @precision shows how to refine from
Rust yourself.

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`refinement_steps`], [`-1`, `0`, or a positive integer], [With `-1`, SPARTA refines until it reaches `refinement_rtol` or stops improving, with a safety cap of 50 steps. With `0`, refinement is off. A positive number is a hard cap.],
  [`refinement_rtol`], [between 0 and 1], [Refinement stops when the relative residual $norm(b - A x) slash norm(b)$ falls below this value.],
  [`refinement_minimize`], [bool], [Scales each correction to shrink the residual as much as possible. Each step costs one extra matrix-vector product on the CPU. On badly conditioned matrices, it roughly halves the number of steps.],
  [`equilibrate`], [bool], [Scales rows and columns to similar sizes before the values are rounded to single precision. When entries differ by many orders of magnitude, this can improve accuracy. It costs extra memory and CPU time.],
  [`safe_mode`], [bool], [Checks every new matrix passed to `factorize`. When it is off, a matrix stored exactly like the analyzed one is read straight into the analyzed layout. That is much faster, but its values are not checked for infinities or NaNs. In both cases, SPARTA detects a different pattern and analyzes it again.],
  [`verbose`], [bool], [Prints a short summary after each solve: method, ordering, fill-in, memory, timings and residuals. Measuring the residuals adds some CPU time.],
)

== Ordering settings

The ordering decides the fill-in, and so most of the memory and time.
SCALPEL, SPARTA's own nested dissection, recursively cuts the matrix into two
halves joined by a small *separator*. It runs on the threads that `threads`
sets.

When you factor one pattern many times, as in a frequency sweep, a slower but
better ordering pays off. A faster ordering does not always give a faster
factorization, so time the whole workload.

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`imbalance`], [0.5 to 0.9], [Sets the largest share of the matrix allowed in one half of a split. Near 0.5, the halves must be even. Larger values give more freedom to find a small separator, which can reduce fill-in.],
  [`wide_cycles`], [0 or more], [Sets the cycles that widen a separator and search for a smaller one. With `0`, the ordering is faster, but the factors get more fill-in.],
  [`fm_passes`], [0 or more], [Sets the Fiduccia–Mattheyses passes, a refinement that may accept a temporary worsening to find a better separator. With `0`, the ordering is much faster, but the factors get much more fill-in.],
  [`separator_trials`], [1 or more], [Sets how many starting separators are tried per split. SPARTA keeps the best one. More trials help on irregular meshes.],
)

*More SCALPEL options, Rust only.* `config.scalpel` has further fields for
ordering research. Change them only for that research.

#table(columns: (auto, auto, auto, 1fr),
  head([Field], [Type], [Default], [What it does]),
  [`leaf_size`], [`usize`], [`32`], [Parts at most this size (1 to 256) are ordered by exact minimum degree instead of being cut further.],
  [`parallel_threshold`], [`usize`], [`4096`], [Parts at least this size may be processed in parallel.],
  [`spectral_candidates`], [`usize`], [`4`], [Extra starting separators cut across the graph's smoothest vibration modes. They give flatter cuts and a shallower tree, which factors and solves faster on the GPU. `0` orders faster.],
  [`refine_all_candidates`], [`bool`], [`true`], [Refine every candidate separator before choosing one.],
  [`half_levels`], [`bool`], [`true`], [Build starting separators from distances to two ends of the graph.],
  [`balance_quality`], [`bool`], [`true`], [Favour balanced halves as well as small separators.],
  [`refinement_passes`], [`usize`], [`4`], [Greedy improvement sweeps per level. `0` skips them.],
  [`one_sided_refinement`], [`bool`], [`false`], [Move vertices only towards the smaller half.],
  [`coarsen_threshold`], [`usize`], [`512`], [Parts above this size are simplified before cutting, then refined. `0` disables; otherwise at least 32.],
  [`compress`], [`bool`], [`true`], [Merge vertices with identical neighbours before ordering.],
  [`cover_refinement`], [`bool`], [`true`], [Improve separators with a vertex-cover search at their boundaries.],
  [`fine_selection`], [`bool`], [`false`], [Choose among candidates on the full graph instead of the coarsest one. More ordering work.],
)

== Front settings

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`nemin`], [1 or more], [Sets how many columns a supernode must have before it may merge with its neighbour. Larger values give fewer, wider fronts. These run more efficiently on the GPU, but use more fill-in and memory. When many small fronts take most of the time, try 32.],
  [`max_front`], [1 or more, or unlimited], [Makes the analysis fail when a front would have more rows than this, before any GPU memory is claimed. Use it to reject a pathological matrix early. It does not limit the total memory.],
  [`small_front_rows`], [1 or more], [Fronts up to this height use a serial GPU kernel. Taller fronts use cooperative kernels. Which is faster depends on the front sizes.],
)

== Numerical settings <numerics>

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`force_cpu`], [bool], [Factors and solves on the CPU in double precision instead of on the GPU. This is much slower, but accurate to about sixteen digits. See @precision.],
  [`hybrid_cpu`], [0 to below 1], [Lets the CPU do this share of the factorization while the GPU does the rest. `0` uses the GPU alone. The value is only a starting point: SPARTA adapts the share after every factorization. In Python, `hybrid=True` sets 30%. Only used on Metal. See @hybrid.],
  [`lu_diag_threshold`], [0 to 1], [Only affects LU factorization. With `0`, each pivot is the largest candidate in its column (strict partial pivoting). Above 0, SPARTA keeps the diagonal entry as pivot if it is at least this fraction of the largest candidate. For example, `0.1` accepts a diagonal one tenth as large. This saves row swaps but accepts weaker pivots, and *changes the results*. If you use it, check the residuals.],
  [`static_pivot`], [0 to below 1], [Only affects LDLᵀ factorization. SPARTA keeps each pivot in its front, so a pivot can be tiny without being zero. This happens, for example, with zero diagonal entries in saddle-point or constrained systems. Dividing by such a pivot can ruin the factorization. SPARTA therefore raises any pivot smaller than `static_pivot` times the largest matrix entry to that size, and counts it in `perturbed_pivots`. In CPU mode, the raise is also capped at `static_pivot` times the largest entry below the pivot, so a small pivot on a badly scaled diagonal is left alone. In GPU mode, single precision cannot tell such a pivot from rounding noise, so the threshold applies as is. The default, `1e-12`, is far below the pivots of ordinary matrices. With `0`, SPARTA replaces exact zeros only, as before version 0.3.5. LU and Vulkan ignore it. Only available in Python and Rust. In C, Julia and MATLAB, `SPARTA_STATIC_PIVOT` overrides the default.],
  [`supernodal`], [bool], [Only affects LDLᵀ factorization. In CPU mode, it factors each block of columns in place instead of in a separate frontal matrix, which makes the factorization 25 to 43% faster and lowers peak memory. In GPU mode, it halves the work of setting up each front, which makes the factorization about 15% faster. It is expected to be strictly better than the multifrontal factorization, which is used when it is turned off: the pivots and the accuracy are the same, but the two can differ slightly in how they round. One exception is possible in CPU mode. Threads that update the same rows of the factor take turns, so on a matrix where many independent parts feed into the same rows, the threads can end up waiting on each other. None of our FEM test cases shows this. If the CPU factorization of your matrix is slower than expected, turn it off to check whether this is the cause. LU, the hybrid's CPU share and Vulkan ignore it.],
  [`real_split`], [bool], [Only affects complex LDLᵀ factorization. It factors the parts of the matrix without imaginary entries in real arithmetic. In CPU mode, this makes the factorization 1.1 to 1.8 times faster and lowers peak memory by 12 to 28%, but only when `supernodal` is on. In GPU mode, the solver runs its second factorization without it and its third with it, then keeps whichever was faster, so the first three factorizations can differ in time. The results are equally accurate either way, so turn it off only if those first factorizations must all take the same time.],
)

== CPU backend settings

These settings only apply in CPU mode (`force_cpu`).

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`cpu_panel`], [1 or more], [Sets how many pivots are gathered before a trailing update. It applies to LU, and to the $L D L^T$ fronts that the blocked route cannot take (an exactly zero or tiny pivot).],
  [`cpu_block_rows`], [1 or more], [Sets the rows per block of the trailing update. When one front is split across threads, this is also the unit of work. Smaller blocks waste less work, but add overhead.],
  [`cpu_blas_min_rows`], [1 or more], [Blocks with fewer trailing rows than this skip the dense kernels. Raise it to avoid the call overhead on small blocks.],
)

== Threads

`threads` sets how many CPU threads SPARTA uses for the analysis, the CPU
factorization and solves, and the hybrid's CPU share, on every backend.

#table(columns: (auto, 1fr),
  head([Value], [What happens]),
  [`0` (default)], [SPARTA shares the process-wide pool, one thread per core unless `RAYON_NUM_THREADS` sets it before the program starts.],
  [1 to 4096], [Each analysis and solver gets a pool of exactly that many threads; the dense kernels run on the same pool.],
)

Set it when SPARTA shares the machine. For example, when other work runs in
the same process, such as a simulation assembling the next matrix, or when
several solvers factor at once. `SPARTA_NUM_THREADS` overrides it from
outside, in every language.

== GPU settings (advanced)

Change these only when `sparta.optimize` or your own measurements show it
helps.

#table(columns: (auto, auto, 1fr),
  head([Setting], [Values], [What it does]),
  [`schur_batch`], [0 or more], [Sets how many pivot panels complete between update rounds for the rest of the front. With `0`, there is one round per level of the tree.],
  [`panel_threads`], [`0`, or 32 to 1024], [Sets the threads per group in the panel factorization. With `0`, SPARTA uses the built-in 512.],
  [`solve_threads`], [`0`, or 32 to 1024], [Sets the threads per front in the solves. With `0`, SPARTA chooses per front. Values up to 256 select the 256-thread path. Larger values select the 512-thread path.],
  [`panel`], [16], [Sets the pivots per panel. Other widths need kernel changes.],
  [`schur_block`], [1 to 8], [Sets the tile size in the main update. Each group computes a `16 × schur_block` square tile.],
  [`strip_block`], [1 to 8], [Sets the tile height in the strip update: `16 × strip_block` rows, which must be a multiple of `panel`.],
  [`narrow_cols`], [1 or more], [Fronts wider than this take the wide-front kernels.],
  [`pivot_threads`], [multiple of 32, up to 1024], [Sets the threads that search for one pivot together.],
  [`schur_chunk`], [1 or more], [Sets the columns staged together per update pass.],
)

The last six settings are compiled into the GPU kernels. The first time a
process creates a solver with a new combination, compilation takes about
300 ms. Both update kernels must also fit the GPU's shared memory per group:
32 KiB on Metal, and at least 16 KiB on Vulkan.

- main update: $2 times (16 times "schur_block") times "schur_chunk" times 8$ bytes,
- strip update: $(16 times "strip_block" + 16) times "schur_chunk" times 8$ bytes.

If a combination breaks a limit, SPARTA rejects it with a message that names
the setting.

== Environment variables

#table(columns: (auto, 1fr),
  head([Variable], [Effect]),
  [`SPARTA_SCHUR_BATCH`], [Overrides the default of `schur_batch`.],
  [`SPARTA_PANEL_THREADS`], [Overrides the default of `panel_threads`.],
  [`SPARTA_SOLVE_THREADS`], [Overrides the default of `solve_threads`.],
  [`SPARTA_LU_DIAG_THRESHOLD`], [Overrides the default of `lu_diag_threshold`.],
  [`SPARTA_STATIC_PIVOT`], [Overrides the default of `static_pivot`.],
  [`SPARTA_HYBRID_CPU`], [Overrides the default of `hybrid_cpu`.],
  [`RAYON_NUM_THREADS`], [Sizes the process-wide thread pool that SPARTA uses when `threads` is `0`. Set it before the program starts.],
  [`SPARTA_NUM_THREADS`], [A positive count overrides `threads` wherever it was given, in every language.],
  [`SPARTA_KERNEL`], [Forces a CPU dense kernel set: `sme512`, `avx512`, `avx2`, `neon` or `scalar`, or `accelerate` on a Mac (@blas).],
  [`SPARTA_PROFILE`], [`1` prints the time of each stage of analysis and factorization; `2` adds GPU dispatch detail.],
)

The first six variables change only the defaults, so a value you set in code
wins. `SPARTA_NUM_THREADS` works the other way: it wins over code.

= Precision <precision>

== Why answers are not exact

*Rounding.* The GPU computes in single precision, which keeps about seven
significant digits. Every operation rounds, and a factorization performs
billions of operations. The CPU backend uses double precision, about sixteen
digits.

*Conditioning.* Some systems amplify small errors. The *condition number* of
$A$ says by how much. With a condition number of $10^6$, rounding errors in the
seventh digit can grow into errors in the first. FEM matrices at low
frequencies, or with very fine meshes, are often badly conditioned.

*Residual and error are different.* The *relative residual*
$norm(b - A x) slash norm(b)$ measures how well $x$ satisfies the equations. It
is cheap to compute. The *error* $norm(x - x_"true") slash norm(x_"true")$ is
what you care about, but you cannot compute it without the true answer. On a
badly conditioned matrix, the error can be far larger than the residual. On
one of our test cases (condition number about $2.4 times 10^9$), a residual of
$4 times 10^(-7)$ came with an error of $8 times 10^(-5)$, two hundred times
larger.

#note[*Ask for a tighter residual than the accuracy you need.* On that system,
an error of $2 times 10^(-8)$ needed `refinement_rtol = 1e-10`.]

== The options at a glance

#table(columns: (auto, auto, auto, auto),
  head([Approach], [Accuracy], [Cost], [Available in]),
  [GPU, no refinement], [residual about $10^(-5)$], [fastest], [Rust, Python],
  [GPU with refinement], [residual $10^(-6)$ by default, or as tight as you ask], [about 9% more time on large systems at the default], [Python built in; Rust with @rust-refine],
  [CPU backend], [double precision, about 16 digits], [much slower], [Rust, Python],
)

*Iterative refinement* keeps the single-precision factors and repeats three
steps: compute the residual $r = b - A x$ in double precision, solve $A d = r$
with the factors, and add $d$ to $x$. Each round gains several digits.

== What to do: step by step

+ *Keep your data in double precision.* Pass $A$ and $b$ as float64 if they
  are real, and as complex128 if they are complex. Refinement measures the
  residual against the values you pass, so it cannot recover digits lost
  before SPARTA saw them.
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

*Python.* Right after `factorize()`, before any solve, `solver.quality` says
how close the factorization came to breaking down. Reading it costs no extra
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
  [`degraded`], [Some pivots are close to single-precision resolution. Solves are still usable, but use refinement and check the result.],
  [`unusable`], [The factorization effectively broke down. Refinement cannot recover it, so use the CPU backend.],
  [`unknown`], [There is not enough information to judge. This happens on the CPU backend before any refined solve. Check the refinement report or the residual instead.],
)

*Rust.* Two methods give the raw signals:

#code(```rust
let (smallest, largest) = solver.pivot_range().unwrap();
let nudged = solver.perturbed_pivots();
```)

- `pivot_range()` returns the smallest and largest pivot size. A ratio
  `smallest / largest` near single-precision resolution (about $10^(-7)$) is a
  warning. Far below it, expect the solve to fail.
- `perturbed_pivots()` counts the pivots SPARTA had to raise, away from zero or
  up to `static_pivot`, to avoid a breakdown. A nonzero count means the
  factors belong to a slightly different matrix, so use refinement.

Neither a good verdict nor zero perturbed pivots guarantees an accurate
answer. They catch breakdowns. The residual confirms the result.

== Getting more accuracy from the GPU <more-accuracy>

=== In Python

Refinement is on by default, with a target residual of $10^(-6)$:

#code(```python
# Keep double precision: float64 for a real system ...
A = A.astype(np.float64)
b = b.astype(np.float64)
# ... or complex128 for a complex one.
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

A refined solve always returns double precision: float64 when the matrix and
b are both real, complex128 otherwise. Each report contains:

#table(columns: (auto, 1fr),
  head([Field], [Meaning]),
  [`status`], [`CONVERGED`: the target was reached. `STAGNATED`: corrections stopped helping, and the best answer found is returned. `ITERATION_LIMIT`: `refinement_steps` ran out. `NONFINITE`: the solve produced infinities or NaNs.],
  [`corrections`], [How many corrections were applied.],
  [`relative_residual`], [The final relative residual.],
  [`estimated_relative_error`], [An estimate of the error in $x$ itself, from the size of the last correction. This is usually the number you care about. While refinement converges, the estimate is on the safe side. Once refinement stagnates, it is unreliable.],
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

Rust solves return the single-precision answer, so refinement is up to you.
The function below implements it. It needs a way to multiply your matrix by a
vector in double precision, which you pass as `matvec`.

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

The function is written for a complex matrix. For a real one, the same steps
work with `f64` in place of `Complex64`, `f32` in place of `Complex32`, and
`solver.solve_values(&rhs, 1)` in place of `solver.solve(&rhs)`.

Call it after `factor`:

#code(```rust
solver.factor(&values)?;
let x = solve_refined(&mut solver, |x| my_matrix.multiply(x), &b, 1e-10, 20)?;
```)

Compute the residual from your original double-precision matrix, not from
the single-precision values you passed to `factor`. The extra digits come from
there. If the residual stops shrinking, stop refining and switch to the CPU
backend.

== Solving on the CPU in double precision <cpu>

`force_cpu` runs the factorization and the solves on the CPU in double
precision, with the same analysis. It is much slower than the GPU, but keeps
about sixteen digits throughout. Use it for systems too badly conditioned for
refinement. SPARTA never switches to the CPU by itself: if the GPU fails, you
get an error.

*Python*

#code(```python
solver = sparta.analyze(A, force_cpu=True, refinement_steps=0)
solver.factorize()
x = solver.solve(b)
solver.compute_dtype   # float64 for a real A, complex128 for a complex one

# Or switch an existing solver, keeping its analysis, and back again.
solver.factorize(force_cpu=True)
solver.factorize(force_cpu=False)
```)

Refinement also works on the CPU backend, but you rarely need it there.

*Rust.* Set `force_cpu` and pass double-precision values, `f64` or
`Complex64`:

#code(```rust
let config = Config { force_cpu: true, ..Config::default() };
let mut solver = Solver::new(Analysis::ldlt(n, &col_ptr, &row_idx, &config)?)?;
// values and b are &[f64] or &[Complex64]; the last argument counts the b's.
solver.factor_values(&values)?;
let x = solver.solve_values(&b, 1)?;
```)

The large dense operations run on SPARTA's own dense kernels (@blas).

== CPU and GPU together <hybrid>

On a Mac, the CPU and the GPU share one memory, so they can factor one matrix
together. With `hybrid` on, the CPU factors whole branches of the elimination
tree while the GPU factors the rest. The GPU then finishes from where the two
meet.

The CPU computes in the GPU's precision, so the answers match a GPU-only
factorization up to single-precision rounding. It works for LDLᵀ and LU, real
and complex.

#code(```python
solver = sparta.analyze(A, hybrid=True).factorize()   # the CPU starts at 30% of the work
solver.factorize(A2, hybrid=0.2)                      # or start it at 20%
solver.factorize(A3, hybrid=False)                    # the GPU alone again
solver.hybrid_timing                                  # how the last factorization split
```)

In Rust, set `hybrid_cpu` in the `Config`. In C, set `config.hybrid_cpu`
before `sparta_analyze`. In Julia, pass `hybrid = true`.

- *When it starts.* The first factorization already runs split. After that,
  SPARTA adapts the split to the measured speed of each device. In a sweep,
  the split settles from the third factorization on. If you change `hybrid` in
  `factorize`, SPARTA rebuilds the GPU factors.
- *Where it applies.* Only the Metal backend has a hybrid path. With
  `force_cpu`, on Vulkan, or without a GPU, SPARTA ignores the setting.
- *What it gains.* Measured on an M4 Pro with `RAYON_NUM_THREADS=10` (the
  performance cores): 1.37× on an 887k-unknown LDLᵀ system (1.49 s to
  1.10 s), 1.25× at 457k and 1.10× at 109k. For LU, the CPU is slower relative
  to the GPU, so LU gains less: about 1.07× at 887k.

== When refinement does not converge

#table(columns: (1fr, 1fr),
  head([What you see], [What to do]),
  [`status` is `STAGNATED` with a residual above the target], [Try `equilibrate=True`. Then the `careful` preset. Then `force_cpu=True`.],
  [`status` is `ITERATION_LIMIT`], [Raise `refinement_steps`, or set it to `-1`.],
  [`status` is `NONFINITE`, or `quality.verdict` is `unusable`], [The single-precision factorization broke down. Use `force_cpu=True`.],
  [Residual is small but the answer is still inaccurate], [The matrix is badly conditioned. Tighten `refinement_rtol` by two or three orders of magnitude, and read `estimated_relative_error`.],
  [`perturbed_pivots` is nonzero], [Use refinement and check the residual. If refinement stagnates, switch to the CPU backend.],
  [You set `lu_diag_threshold` above 0], [Set it back to `0`. Strict pivoting is the most stable.],
)

== Checking the final answer

Whatever route you take, the residual in double precision is the final check:

#code(```python
residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
```)

In Rust, compute $norm(b - A x) slash norm(b)$ with your own double-precision
matrix-vector product. The most assurance a solver can give is a small residual
together with either a good `quality` verdict, or a converged refinement
report with a small `estimated_relative_error`.
