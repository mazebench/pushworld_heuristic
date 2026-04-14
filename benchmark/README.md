# Translated PushWorld Benchmark

This folder contains the `.txt` translations of every level in `og_benchmark`,
using the same subfolders and filenames.

Cell encoding:

- `..` empty
- `#.` wall
- `P.` player
- `a.` box `a`
- `.a` target `a`
- `ba` box `b` resting on target `a`

DeepMind gate cells (`AW`) are preserved as:

- `.|` empty gate
- `a|` box `a` resting on a gate

That keeps the original gate information instead of flattening it into a wall or
empty floor.
