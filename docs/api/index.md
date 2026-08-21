# API reference

The C reference is generated from the public headers in `include/sif/` by
[Hawkmoth](https://hawkmoth.readthedocs.io/), which parses them with libclang
and emits native Sphinx C-domain entries. There is no separate API site: these
pages carry the same theme, navigation and search as the rest of the
documentation, and a type named here can be linked to from anywhere in it.

One page per header, grouped by subsystem, mirroring the include tree.

```{toctree}
:maxdepth: 2

core/index
structures/index
finder/index
measure/index
model/index
io/index
utils/index
python
```

The pages under each subsystem are generated at build time from the contents
of `include/sif`, one per header, by `docs/_ext/apitree.py`. Adding a header
adds a page; there is no index to keep in step by hand.

## Writing headers for this

Comments are read in Doxygen style -- `@brief`, `@param`, `@return`, `@note`
all work as they always have, through `hawkmoth.ext.javadoc`. Nothing about
the existing headers had to change to produce these pages.

What is new is that the *body* of a comment is reStructuredText, which asks
for four things. The build runs with `-W`, so breaking any of them fails CI
rather than quietly rendering the wrong thing.

Close a multi-line comment on its own line
: ```c
  /** Right: the closer gets a line to itself.
   * Continuation.
   */

  /** Wrong: the closer shares the last text line.
   * Continuation. */
  ```
  In the second form hawkmoth leaves the leading `*` on the continuation
  lines, and reST reads the result as an indented block. A comment that fits
  on one line is fine either way.

Bars are markup
: reST reads `|x|` as a substitution reference, not an absolute value. Put
  the expression in backticks -- `` `nu_t = |delta| / sigma_0(R)` `` -- which
  is where a formula belongs anyway.

Indentation is markup
: An indented run of lines is a block quote, so indent only when you mean a
  definition list or a literal block. Aligning a continuation line under the
  text above it is what breaks.

No Markdown tables
: reST has no `| a | b |` table syntax. A definition list -- term on one
  line, meaning indented under it, blank line between entries -- says the
  same thing and reads better in the source.

Single backticks are literals here, not title references: `default_role` is
set to `code` in `conf.py`, so the existing `` `identifier` `` style renders
as you would expect.
