# Decathlon

This repository contains research work for Mikael Myyrä's master's thesis,
using Jukka Räbinä's mesh generator and Discrete Exterior Calculus solver [gfd]
and prototyped using the similar Python library [PyDEC] by Nathan Bell and Anil
N. Hirani.

## Contents

- DecExercise
  - Practicing using the DEC versions of differential operators. Thanks to
    Jonni Lohi for the example. (documented in Finnish)
- More to come later

## Running the code

For the C++ programs, install GNU Make, GCC, and libopenmpi either by running
`nix-shell` or using your package manager of choice, then navigate to a code
directory and run `make run`. Images will appear in the `build` directory and
text outputs in your console.

For the Python programs you'll need the `gmsh` tool, a few Python libraries, a
whole bunch of C libraries, and `./pydec` added to your `PYTHONPATH`. See
`shell.nix` for details.

Remember to run `git submodule init && git submodule update` after cloning the
repo to get the DEC libraries.

[gfd]: https://github.com/juolrabi/gfd
[pydec]: https://github.com/hirani/pydec
