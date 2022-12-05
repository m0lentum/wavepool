{ sources ? import ./nix/sources.nix
, pkgs ? import sources.nixpkgs { }
}:

let
  gmshPy = pkgs.python3.pkgs.buildPythonPackage rec {
    pname = "gmsh";
    version = "4.11.0";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/ef/65/037b7834acf333644e686fc7e6bb91385cb6b7e7ac324fea2b4a6c6ef406/gmsh-4.11.0-py2.py3-none-manylinux1_x86_64.whl";
      sha256 = "cntqN1k7KzOwgPs1GPN66asbtbsRrdNYYG9KylmBx9s=";
    };
  };
  python = pkgs.python3.withPackages (ps: with ps; [
    numpy
    scipy
    matplotlib
    gmshPy
  ]);
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    niv
    gnumake
    # C++ and dependencies
    gcc
    openmpi
    # Python and dependencies
    python
    gmsh
  ];
  # add pydec from this repo (which was too cumbersome to install with nix)
  # to the python search path
  PYTHONPATH = builtins.toString ./pydec;
  # bunch of dynamically linked libs for gmsh
  LD_LIBRARY_PATH = with pkgs.xorg; with pkgs.lib.strings;
    concatStrings (intersperse ":" [
      "${pkgs.libGLU}/lib"
      "${pkgs.libglvnd}/lib"
      "${pkgs.fontconfig.lib}/lib"
      "${libX11}/lib"
      "${libXrender}/lib"
      "${libXcursor}/lib"
      "${libXfixes}/lib"
      "${libXft}/lib"
      "${libXinerama}/lib"
    ]);
}
