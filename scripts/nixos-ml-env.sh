#!/usr/bin/env bash
# NixOS only: let prebuilt ML wheels (numpy, torch) find libz and libstdc++, and stop a global
# Nix PYTHONPATH from shadowing the project venv. Works from bash or zsh:
#   source scripts/nixos-ml-env.sh
_engram_libs=$(nix build --no-link --print-out-paths nixpkgs#zlib.out nixpkgs#stdenv.cc.cc.lib \
  | sed 's#$#/lib#' | paste -sd: -)
export LD_LIBRARY_PATH="${_engram_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONPATH _engram_libs
