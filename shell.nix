{ pkgs ? import <nixpkgs> { } }:
with pkgs; mkShell {
  name = "ml-friendly-shell";
  buildInputs = [
    glib
    libGL
    glibc
    zlib
    python39
  ];
  shellHook = ''
    # Setup python virtual environment
    python -m venv ~/.venv
    source ~/.venv/bin/activate

    # Standard environment
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib"

    # [CV2] Software / Hardware bridge
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${libGL}/lib"

    # [CV2] Useful data types
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.glib.out}/lib"

    # [Torch (CUDA)] OpenGL
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver-32/lib"

    # [Numpy] Data compression
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.zlib.outPath}/lib"
  '';
}