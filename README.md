# Decagon

This repository contains research work for Mikael Myyrä's master's thesis,
using Jukka Räbinä's mesh generator and Discrete Exterior Calculus solver [gfd].
No 10-sided polygons here, I just can't name anything without making it a pun somehow.

## Contents

- DecExercise
  - Practicing using the DEC versions of differential operators. Thanks to
    Jonni Lohi for the example. (documented in Finnish)
- More to come later

## Running the code

Install GNU Make, GCC, and libopenmpi either by running `nix-shell` or using
your package manager of choice, then navigate to a code directory and run
`make run`. Images will appear in the `build` directory and text outputs in
your console.

Remember to run `git submodule init && git submodule update` after cloning the
repo to get the GFD library!

[gfd]: https://github.com/juolrabi/gfd
