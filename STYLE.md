# sif style guide

Conventions for the sif source tree. They exist so the library reads as though
one person wrote it in one sitting, and so that a name tells you what a symbol
is and where it lives before you go looking.

Rules here are binding for `include/`, `src/`, `python/src/` and `tests/`.

`vendor/` is exempt. Third-party code stays as close to upstream as possible so
it can be re-imported without a merge, which means upstream's naming wins even
where it breaks the rules below. Any local modification must be marked with a
`sif:` comment at the point of change and summarised in a block at the top of
the file, so the divergence from upstream is never silent.

---

## 1. Formatting

Formatting is specified by `.clang-format`. Run it before committing:

```bash
clang-format -i $(git ls-files '*.c' '*.h' | grep -v '^vendor/')
```

Source files are **ASCII only** -- write `--` rather than an em dash, and
spell out Greek letters (`sigma`, `gamma`) in comments. The compiler will
accept UTF-8 in a comment, but not every toolchain agrees on the source
character set, and a stray non-ASCII byte is a miserable thing to debug.

General rules: C99, 2-space indent, 80-column limit, `UseTab: Never`,
attached braces, pointers bound to the type (`sif_real* p`, not `sif_real *p`),
preprocessor directives indented after the hash.

Three things clang-format cannot decide for you:

- **Braces on single-statement bodies.** Optional, but be consistent within a
  function. Always brace a body that spans more than one line, and always brace
  the arms of an `if`/`else` chain if any one of them needs braces.
- **Blank lines.** One blank line between logical steps inside a function; two
  between top-level definitions.
- **Line breaks in expressions.** Break at the lowest-precedence operator and
  indent the continuation so the operands line up with each other, not with the
  opening parenthesis.

---

## 2. Naming

Everything is `snake_case`. Types, functions, variables, files. Macros are the
sole exception and are `SCREAMING_SNAKE_CASE`.

### 2.1 Public functions

```
sif_<domain>_<thing>[_<qualifier>]
```

**`domain`** is the concrete thing the function belongs to — a data structure
(`field`, `grid`, `octree`, `catalog`, `chain_mesh`, `bitmask`), a formalism
(`bbks`, `ep`, `svdw`), or a subsystem (`finder`, `profiles`, `io`,
`prng`, `timer`). It is *not* the directory: the header lives in
`include/sif/structures/field.h`, but the domain is `field`, not
`structures_field`.

Where a physical quantity is defined by a particular formalism, **the formalism
is the domain**, and the quantity is the thing:

```c
sif_bbks_gamma            sif_ep_multiplicity_function
sif_bbks_r_star           sif_ep_first_crossing_counts
sif_bbks_size_function    sif_ep_barrier_smt
```

**`thing`** is the noun the function is about. **`qualifier`**, when present,
names *the input it works from* or *which variant this is* — never the
operation:

```c
sif_size_function_catalog     /* built from a catalog       */
sif_delta_moments_pk          /* built from a power spectrum */
sif_chain_mesh_find_nearest_pbc   /* periodic variant        */
sif_bitmask_set_atomic            /* atomic variant          */
```

In some very specific cases, the **`domain`** attribute should be used as a
**`qualifier`**; specifically, this happens when a **`thing`** is also a 
**`domain`**. For example, `size_function` would generally be a **`thing`**, 
but it assumes the role of **`domain`** because of the `size_function_t` 
object. This makes `size_function` a **`domain`** across the whole library,
thus a framework that computes a size function needs to be specified as
**`qualifier`** instead of **`domain`**. Here is a practical example:
```c
// svdw is correctly a domain here. If somewhere else a 
// multiplicity_function domain were to appear, this function would need
// a renaming to sif_multiplicity_function_svdw
sif_svdw_multiplicity_function 

// svdw decays to a qualifier because the thing is a size function,
// which is a domain elsewhere in the library
sif_size_function_svdw 
```

### 2.2 Verbs

Verbs are not banned; redundant ones are.

- **A function that returns a result names the result, not the act of
  producing it.** The return value already implies computation, so `compute_`,
  `build_`, `calculate_`, `get_` (on a pure query) and `do_` add nothing:
  `sif_build_tessellation_delaunay` → `sif_tessellation_delaunay`.
- **A function that mutates state, or performs an action with an effect, keeps
  its verb.** `sif_catalog_append`, `sif_field_sort_morton`,
  `sif_field_wrap_periodic`, `sif_bitmask_set`, `sif_grid_write`.

Established verbs to prefer over synonyms, so the same idea always has the same
name:

| verb | meaning |
|---|---|
| `_alloc` / `_free` | allocate and release an owning object |
| `_read` / `_write` | move data across the process boundary (files) |
| `_read_into` | read into storage the caller already owns |
| `_reserve` | make room without changing the logical size |
| `_require_` | ensure a precondition holds, recomputing only if stale |
| `_find_` / `_search_` | locate one item / collect many |
| `_insert` / `_append` | add to a structure |

**Accessors are the exception to the redundant-verb rule.** A symmetric
get/set pair reads worse without the verb — `sif_setting(key, fallback)` beside
`sif_setting_set(key, value)` is confusing, not concise. So accessors keep
`get`/`set`, and they always go in **the same fixed place: the last component of
the name**, after any qualifier.

```c
sif_setting_get       /* not sif_get_setting  */
sif_setting_set
sif_size_function_get
```

This only applies to genuine accessor pairs — a plain query that computes
something is not an accessor and drops the verb (`sif_array_sum`, not
`sif_array_sum_get`).

**`to_` marks an in-place conversion.** A function that rewrites its argument
as a different quantity is neither a query nor an ordinary mutation, and
naming it after the result alone would read like a getter:
`sif_grid_to_density_contrast()` overwrites the grid's cell masses with the
density contrast, and the `to_` says so.

### 2.3 Internal functions

**A `sif_` prefix in a definition means the symbol is exported. No exceptions.**
That makes the library's actual surface greppable, which is the whole point.

- **Used in one `.c` file** (internal helper) → `static`, **no prefix**:
  `density_profiles_alloc`, `bin_of`, `ctx_init`.
- **Crosses translation units** through a private header in `src/` →
  `sif__<domain>_<thing>`, with the infix double underscore:
  `sif__field_reserve_block`, `sif__log_impl`.
- **A `static inline` helper defined in a *public* header** → also
  `sif__<domain>_<thing>`. It exports no symbol, but it does occupy the
  namespace of everyone who includes the header, so it needs the prefix; the
  double underscore then says it is not part of the API:
  `sif__splitmix64`, `sif__rotl` in `utils/random.h`.

The same applies to a function that is not `static` but exists only to back a
public macro — `sif__log_impl` is called from user code through `SIF_LOG_INFO`,
so it is part of the ABI without being part of the API.

### 2.4 Reserved identifiers 

C reserves whole classes of identifier for the implementation (C11 §7.1.3), and
using them is undefined behaviour even when it happens to work today:

| pattern | C guideline |
|---|---|
| `__foo`, `__FOO` | reserved **always, in every scope** |
| `_Foo` (underscore + uppercase) | reserved **always, in every scope** |
| `_foo` (underscore + lowercase) | reserved **at file scope** — which is where functions and globals live |

So `__sif_log_impl`, `_sif_log_impl` and `__SIF_CACHE_LINE` are all off-limits.
The infix form `sif__log_impl` and `SIF__CACHE_LINE` reads the same, sorts
beside the public name, and is legal.

**One exception:** struct *members* are in their own namespace and are not file
scope, so a single leading underscore there is fine and is used deliberately to
mark a field callers must not touch:

```c
typedef struct {
  sif_real* _position_block;  /* owned backing store, not for callers */
  sif_real* x;
} sif_field_t;
```

Never use `__` on a member either — that one is reserved everywhere.

### 2.5 Macros

| kind | form | example |
|---|---|---|
| public | `SIF_<DOMAIN>_<THING>` | `SIF_FINDER_MESH_MAX_CELLS` |
| public, no domain | `SIF_<THING>` | `SIF_OK`, `SIF_REAL_ABS`|
| internal, crosses TUs | `SIF__<DOMAIN>_<THING>` | `SIF__DELTA_TAG` |
| file-local, inside a `.c` | short and unprefixed | `WRAP_PBC` |

**Anything reachable from a public header must carry `SIF_`.** A public header is
included into somebody else's translation unit, and an unprefixed `NODISCARD`,
`MIN` or `REAL_MAX` sitting there is a collision waiting to happen in code that
has nothing to do with sif. This applies to helper attributes and math wrappers
as much as to constants: `SIF_NODISCARD`, `SIF_HOT_LOOP`, `SIF_ALIGN_T`,
`SIF_REAL_COS`.

File-local macros may be short, but `#undef` them at the end of the file if the
name is generic enough to surprise someone reading further down.

Wrap macro parameters in parentheses, wrap the whole body in parentheses, and
evaluate each parameter exactly once — or make it a `static inline` function,
which is almost always the better answer in C99.

### 2.6 The Python bindings

`python/src/` is the one place the rules above bend, because it has two
audiences. What Python sees follows PEP 8, since that is what a Python user
expects; what C sees follows the object it implements.

**Python-visible names are PEP 8.** Types are `CamelCase` (`Field`, `Grid`,
`ChainMesh`); functions, methods, properties and keyword arguments are
`snake_case` (`sort_morton`, `box_length`, `max_per_leaf`). Names mirror the C
API with the `sif_` prefix dropped, so `sif_grid_to_density_contrast()` is
`Grid.to_density_contrast()`. Where the C name carries its domain, the Python
one keeps it -- `pysif.model.bbks_g` -- so the two APIs remain searchable
against each other.

**Binding-internal C identifiers are named for the Python object**, in
`sif<Type>` form:

| kind | form | example |
|---|---|---|
| instance struct | `sif<Type>Object` | `sifFieldObject` |
| type object | `sif<Type>Type` | `sifFieldType` |
| method or slot | `sif<Type>_<method>` | `sifField_sort_morton` |
| free function | `py_sif_<name>` | `py_sif_read_field` |

This is deliberately unlike the library's own `snake_case`: an identifier in
these files is either glue for a Python type, and looks it, or a call into the
C API, and looks like the rest of sif.

### 2.7 Types

A type declared in a header carries the `sif_` prefix, public or internal.
Types generate no linker symbol, so there is no `sif__` form — an internal type
is simply declared in a private header and looks like any other.

A type confined to a single `.c` follows the same rule as a `static` function
and takes **no prefix** — `particle_sort_t`, `grid_slab_t`. It still takes `_t`
if it is composite.

**The `_t` suffix marks a composite type.** Structs and enums take it; a
typedef that merely renames a scalar does not:

```c
typedef struct { ... } sif_field_t;        /* composite → _t   */
typedef struct { ... } sif_octree_node_t;
typedef enum { ... } sif_filter_type_t;

typedef float sif_real;                    /* scalar alias → no suffix */
typedef uint32_t sif_option;
```

Reading `sif_real` as a bare type name is the point: it appears in nearly every
signature in the library, and the suffix would be noise. It also keeps the type
and its math wrappers spelled the same way (`sif_real` / `SIF_REAL_SQRT`), and
matches the convention other numerical C libraries use for scalar aliases
(`fftw_complex`, `GLfloat`). As a bonus, POSIX reserves `_t` for itself, so not
using it here is the safer choice as well as the tidier one.

Enumerators are macro-style: `SIF_DELTA_FILTER_GAUSSIAN`.

`sif_real` is the library's floating-point type and is `float` or `double`
depending on `SIF_USE_DOUBLE`. **Never write `float` or `double` in library code
where a physical quantity is meant** — use `sif_real`, and reach for the
`SIF_REAL_*` math wrappers rather than `cos`/`sqrt` directly, so a double build
does not silently call the float routine. Fixed-width integers come from
`<stdint.h>`: `uint64_t` for counts and indices that can exceed 4 G, `uint32_t`
for bin and cell indices, `int` only for status codes and small loop counters.

### 2.8 Variables

Short names for short lives, descriptive names for long ones: `i`, `d`, `k` for
loop indices is right; `n_particles`, `half_span`, `inv_side` for anything that
survives more than a few lines. Prefix counts with `n_`. Booleans read as
predicates (`is_sorted`, `has_bounds`). No Hungarian notation, no type suffixes.

### 2.9 Files and include guards

File names are `snake_case` and match the API spelling: the header that declares
`sif_size_function_*` is `size_function.h`, not `sizefunction.h`.

The guard is derived **mechanically from the path**:

| file | guard |
|---|---|
| `include/sif/structures/field.h` | `SIF_STRUCTURES_FIELD_H` |
| `include/sif/model/bbks.h` | `SIF_MODEL_BBKS_H` |
| `src/measure/delta_common.h` | `SIF__MEASURE_DELTA_COMMON_H` |

Public headers take `SIF_`, private headers under `src/` take `SIF__`. No
leading underscores, no trailing `__`.

---

## 3. Headers and includes

`include/sif/` is the public API. `src/**/*.h` is private and may be included
only from within `src/`. If a public header needs something, that something is
public — do not reach into `src/` from `include/`.

**Every header must be self-contained**: it compiles on its own, including
whatever it needs and relying on no particular include order. It must also be
idempotent, hence the guard.

Include order, one blank line between groups (clang-format sorts within a group
but preserves the groups):

```c
#include "sif/measure/profiles.h"   /* 1. this file's own header, first  */

#include "sif/core/macros.h"        /* 2. public sif headers             */
#include "sif/structures/chain_mesh.h"

#include "core/get_system.h"        /* 3. private sif headers            */

#include <math.h>                   /* 4. system and standard headers    */
#include <stdlib.h>
```

Putting the file's own header first is not cosmetic: it is what proves the
header is self-contained, because nothing else has been included yet to cover
for it.

Prefer forward declarations to includes in headers where the type is only used
behind a pointer.

---

## 4. Comments

The library is a scientific tool: someone reading it needs to know *why the
algorithm is what it is*, and that is what comments are for. Comments that
restate the code are noise and get deleted.

Write down the things the code cannot say by itself: the invariant a loop
maintains, the reason a bound is what it is, the paper an expression comes from,
the failure that motivated a guard and so on.

### 4.1 Documentation comments

Every public function, type and macro gets a Doxygen block in the **header**,
using the standard syntax `/**`:

```c
/**
 * @brief One line, one sentence, imperative.
 *
 * The prose that matters: what the caller must guarantee, what is returned,
 * what it costs, which paper the expression comes from. Only if there is
 * something to say and not too long (no wall of text).
 *
 * @param origin Low corner of the bounding cube (center - half_span)
 * @param inv_side 1 / (2 * half_span)
 * @return SIF_OK, or a negative SIF_ERR_* code
 */
```

Document the contract once, in the header. The implementation gets comments
about *how*, not about *what the caller sees*.

Use `@param`, `@return`, `@note`, `@warning`. Do not document a parameter whose
name already says everything — an empty `@param n_bins Number of bins` is
padding.

Where a function returns a pointer, or takes one it keeps, the block must say
who owns the memory afterwards and what invalidates it. That is the part a
caller cannot work out from the signature.

### 4.2 Macro families

A family of related macros — the bits of one option field, a set of status
codes — is documented **as a group**, not one block per macro. Sixty
individually-documented flag bits are unreadable, and the thing worth explaining
is almost always the choice between them rather than any single value.

```c
/**
 * @defgroup delta_shuffle Surrogate field generation
 * @brief How the surrogate field for a PDF comparison is generated.
 *
 * PHASES keeps every |delta_k| and randomizes only the phase, so the realized
 * P(k) is bit-for-bit the input's and any change in the PDF is attributable to
 * phase information alone. GAUSSIAN additionally resamples the amplitudes.
 * @{
 */
#define SIF_DELTA_SHUFFLE_NONE   (0u << 8)
#define SIF_DELTA_SHUFFLE_PHASES (1u << 8)
/** @} */
```

Standalone macros still get their own block.

### 4.3 Implementation comments

Plain `/* ... */` blocks above the code they describe, or short trailing
comments for a single line. Divide a long file with section separators:

```c
/* --- memory --- */
```

Mark unfinished work as `TODO:` or `FIXME:` with enough context to act on. No
commented-out code — that is what git is for.

---

## 5. Error handling and ownership

**Status codes.** A function that can fail in a way the caller may want to
distinguish returns `int`: `SIF_OK` (0) on success, a negative `SIF_ERR_*`
otherwise. Add new codes to `macros.h`; do not invent local conventions and do
not return bare `-1`.

**A file format is the exception, and carries its own status enum.** The rule
above is about *functions*: a caller wants to know that the call failed, and
`SIF_ERR_*` names every reason a function has. A file format is a different
kind of thing — its failures are properties of the file, not of the call, and
"this is not a `.sdf` file", "this one was written by a newer sif" and "this
one is truncated" lead a caller to do three different things while all three
would collapse into `SIF_ERR_IO`. So a format may define its own enum,
`sif_<format>_status_t`, listing them.

Such an enum stays compatible with the library-wide set rather than replacing
it: success is 0, every failure is negative, the generic conditions **alias**
the `SIF_ERR_*` code they correspond to, and the format's own codes start at
`-100` so the generic set keeps room to grow. `!= SIF_OK` and `< 0` therefore
mean what they mean everywhere else. Every such enum comes with a
`sif_<format>_strerror()`.

**Where a subsystem has its own status enum, the status travels as an argument,
not as a return value.** It is the last parameter, it must not be NULL, and the
return value is left free to be the thing the call produces — an open file, a
catalogue — so the API says the same thing everywhere instead of returning a
status here and a pointer there.

Three rules make that worth the extra argument, and they are the whole
convention:

- **An error is inherited.** A call entered with `*status` already set does
  nothing and returns immediately. A sequence of calls therefore needs one
  check at the end rather than one after each, which is the point of the style.
- **The first error wins.** Nothing overwrites a status that is already set,
  so the code that reaches the check is the one that says what actually went
  wrong.
- **Release still runs.** A `_close`, `_free` or equivalent ignores the
  inherited error and cleans up anyway; it may set a status of its own only if
  none is set yet.

Functions that cannot fail — accessors, queries — take no status argument.

**Allocators return pointers**, `NULL` on failure. Every `sif_*_alloc` has a
matching `sif_*_free`, and **`_free` must accept `NULL`** and must be safe on a
partially constructed object — allocators clean up after themselves by calling
their own `_free` on the failure path.

**Mark functions whose result must not be dropped** with `SIF_NODISCARD`. Every
allocator and every status-returning function qualifies.

**Ownership is stated in the header, not inferred.** If a function takes
ownership of a buffer, or returns one the caller must free, the Doxygen block
says so explicitly. Structures own their buffers outright; there is no borrowing
mode, and a function that would need one takes a `_into` variant writing into
caller storage instead (`sif_field_read_into`).

**Validate at the boundary.** Public entry points check their arguments and
return `SIF_ERR_INVALID`; internal helpers assume the contract holds and use
`SIF_ASSERT` for invariants. `SIF_ASSERT` compiles away unless
`SIF_DEBUG_CHECKS` is configured, so it is free in the hot loops — but for the
same reason it must never carry a side effect.

---

## 6. Portability

**A function that takes no arguments is declared `(void)`, never `()`.** In C
the empty parameter list means *unspecified*, not *none*, so `sif_finalize()`
would accept `sif_finalize(1, 2, 3)` without a diagnostic.

C99, no compiler extensions outside `macros.h`. Anything compiler-specific
(`__attribute__`, builtins, pragmas) is wrapped in a `SIF_*` macro there with a
neutral fallback, so the rest of the tree stays clean.

**The target is C99 on POSIX**, and `<unistd.h>`, `<sys/stat.h>` and the rest of
that surface may be used directly. Windows is not a supported platform: a
`#ifdef _WIN32` branch in library code buys a file that compiles on a system
nobody builds or tests sif on, and hides the fact that the neighbouring file
would not. Anything genuinely platform-dependent goes behind a `SIF_*` macro in
`macros.h`, like everything else.

The library is parallel: anything writing to shared state from an OpenMP region
uses the `SIF_ATOMIC_*` wrappers. Do not assume a thread count, and do not let a
result depend on scheduling — a reduction whose answer changes with thread count
is a bug, not a rounding detail.

Note that the release build uses `-ffast-math`, so code that depends on exact
IEEE-754 semantics needs its own compilation flags, as
`vendor/predicates/predicates.c` does in `CMakeLists.txt`.

---

## 7. Checking conformance

Most of this is mechanically greppable, which is the point:

Note that `git grep`'s ERE mode has no `\b` or `\w`; the patterns below use `-P`
or POSIX classes deliberately, so do not "simplify" them.

```bash
# 1. reserved identifiers we own (the filter drops the ones the
#    implementation legitimately provides)
git grep -hoP '(?<![A-Za-z0-9_])(__[A-Za-z]\w*|_[A-Z]\w*)' \
    -- 'include/*' 'src/*' 'tests/*' \
  | grep -vE '^(__attribute__|__GNUC__|__clang__|__VA_ARGS__|__FILE__|__LINE__|__func__|__STDC\w*|__APPLE__|__linux__|__unix__|__x86\w*|__ARM\w*|__aarch64__|__SSE\w*|__AVX\w*|__builtin\w*|__atomic\w*|__ATOMIC\w*|__restrict|__inline|_Atomic|_Bool|_Complex|_Generic|_Pragma|_Static_assert|_Thread_local|_OPENMP|_WIN32|_MSC\w*|_POSIX\w*|_GNU\w*|_FILE_OFFSET\w*|_LARGEFILE\w*)$' \
  | sort -u

# 2. unprefixed macros escaping through public headers
git grep -hoE '^[[:space:]]*#[[:space:]]*define[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' \
    -- 'include/*' \
  | grep -oE '[A-Za-z_][A-Za-z0-9_]*$' | grep -vE '^SIF_|_H__?$' | sort -u

# 3. guards that do not match their path
git ls-files 'include/*.h' 'src/*.h' | while read -r f; do
  want=$(echo "$f" | sed -E 's|^include/sif/|SIF_|; s|^src/|SIF__|' \
    | tr 'a-z/.' 'A-Z__')
  grep -q "$want" "$f" || echo "$f: expected $want"
done

# 4. licence header
for f in $(git ls-files '*.c' '*.h' '*.py' | grep -v '^vendor/'); do
  head -20 "$f" | grep -q SPDX-License-Identifier || echo "no SPDX: $f"
done

# 5. formatting
clang-format --dry-run --Werror $(git ls-files '*.c' '*.h' | grep -v '^vendor/')
```

As of writing, (1) reports 272 identifiers, (2) reports 49 and (3) reports
every header in the tree. That is the size of the job, and each number should
reach zero as the per-file pass proceeds.

---

## 8. Checklist before a file is done

- [ ] Licence header present, `SPDX-License-Identifier: GPL-3.0-or-later`
- [ ] `clang-format` clean
- [ ] No identifier begins with `_` or `__` (struct members excepted)
- [ ] Guard matches the path
- [ ] Public symbols are `sif_<domain>_<thing>`; statics carry no prefix
- [ ] Public macros carry `SIF_`
- [ ] Own header included first, groups separated
- [ ] Public functions have a `/**` block stating contract and ownership
- [ ] Allocator/`_free` pair, `_free` accepts `NULL`, `SIF_NODISCARD` where due
- [ ] `sif_real` and `SIF_REAL_*` used for floating-point work
- [ ] Comments explain why, not what
