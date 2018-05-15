# HGCAL Trigger Geometry Utilities
This is a Python package used to manipulate geometries, like:
* Mapping one geometry to an other
* Finding nearest neighbors

## Installation
First clone the package:
```bash
git clone git@github.com:PFCal-dev/GeometryUtilities.git GeometryUtilities
```

This package depends on several python packages, listed in `requirements.txt`:
* `attrs`
* `scipy`
* `numpy`
* `root-numpy`
* `Shapely`

In general it is better to encapsulate these packages inside a virtual environment (using for instance `virtualenvwrapper`).

It also depends on the [GEOS C++ library](https://trac.osgeo.org/geos/) used by `Shapely`.


On SLC6 lxplus, the default versions of python and ROOT are antique, so a couple of setup commands are needed to have a working environment. The following script should be sourced each time to setup Python, ROOT, and the virtual environment (**outside a CMSSW environment**).
```bash
source init_env_lxplus.sh
```
It will install everything the first time it is called (in particular it will download and compile `GEOS`, which takes some time). The `GEOS` library will be installed in `~/.local`, so `~/.local/lib` should be included in `LD_LIBRARY_PATH`.
Then, once everything is already installed it will just activate the virtual environment.

If it worked correctly the terminal prompt should be prefixed with `(geometry)`, meaning that you are inside the virtual environment. To leave the virtual environment, just type `deactivate`.

On machines other than lxplus, the same thing can be done, just the Python and ROOT setup at the beginning of the script would need to be modified.

## Usage
The package code is located in `geometry`. This code is used from scripts located in the `scripts` directory. The scripts currently available are:
* `produce_cell_mapping_pickle.py`: produce the mapping from the CMSSW geometry to the Zoltan/Split geomeytry
* `produce_panel_mapping_pickle.py`: produce the panels mapping
* `produce_mappings_cmssw.py`: take pickle files from the previous scripts and produce mapping files that can be read by the CMSSW trigger geometry
* `compare_cell_mappings.py`: check the consistency of mapping files, for instance based on different layers
* `produce_cell_neighbors_pickle.py`: produce the map of neighbors for the Zoltan/Split geometry
* `produce_neighbors_cmssw.py`: tacke pickle file from the previous script and produce the corresponding CMSSW-compatible file
* `check_neighbors.py`: performs basic checks on neighbor maps
