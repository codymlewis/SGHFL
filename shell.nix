# let
#   pkgs = import <nixpkgs> {};
# in pkgs.mkShell {
#   packages = [
#     pkgs.python3
#     pkgs.poetry
#     pkgs.zlib
#   ];
#   shellHook = ''
#     poetry install
#     poetry shell
#   '';
# }
{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "smahfl";
  targetPkgs = pkgs: (with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
    zlib
  ]);
  runScript = ''
    #!/usr/bin/env bash
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
    if [[ $(lspci | grep 'NVIDIA') ]]; then
      pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
      pip install jax
    fi
    if [ ! -d data/ ]; then
      cd data_processing/
      python acquire.py
      cd ../../
    fi
    bash
  '';
}).env
