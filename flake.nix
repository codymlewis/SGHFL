{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      nixpkgsFor = forAllSystems(system: import nixpkgs { inherit system; });
    in
    {
      devShells = forAllSystems(system:
        let
          pkgs = nixpkgsFor.${system};
          python-env = pkgs.python312.withPackages (pp: with pp; [
            jax
            jaxlib
            numpy
            pandas
            graphviz
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python312
              python312Packages.pip
              python312Packages.virtualenv
              pciutils
              zlib
              gcc-unwrapped
              pyright
              ruff
              ruff-lsp
              isort
            ];
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
            buildInputs = [ python-env ];
            shellHook = ''
              #!/usr/bin/env bash
              if [ ! -d venv/ ]; then
                virtualenv venv
                source venv/bin/activate
                pip install -r requirements.txt
              else
                source venv/bin/activate
              fi
              if [ ! -d data/ ]; then
                cd data_processing/
                python acquire.py
                cd ../
              fi
            '';
          };
        }
      );
    };
}
