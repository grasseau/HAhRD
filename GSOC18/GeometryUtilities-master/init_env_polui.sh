#! /usr/bin/env bash

# setup Python 2.7 and ROOT 6
source /opt/rh/devtoolset-2/enable
source /opt/exp_soft/llr/root/v6.06.00-el6-gcc48/bin/thisroot.sh
python_path=/opt/exp_soft/llr/python/2.7.10
if [ "x$(which python 2>/dev/null)" != "x"$python_path"/bin/python" ] ; 
then
  export PATH=$python_path/bin:$PATH
  export LD_LIBRARY_PATH=$python_path/lib:$LD_LIBRARY_PATH
fi
unset python_path

# Install geos if not already installed
if [ ! -f ~/.local/lib/libgeos.so ]; then
  wget http://download.osgeo.org/geos/geos-3.6.1.tar.bz2
  tar xvf geos-3.6.1.tar.bz2
  cd geos-3.6.1/
  ./configure --prefix=$HOME/.local
  make
  make install
  cd ..
fi

# Install virtualenvwrapper package if not already installed
if [ ! -f ~/.local/bin/virtualenvwrapper.sh ]; then
  # deleting cached packages to force download
  rm -rf ~/.cache/pip/
  pip2.7 install --user -I virtualenvwrapper
fi

# Create virtual env and install dependencies
if [ ! -d ~/.virtualenvs/geometry/ ]; then
  mkdir -p ~/.virtualenvs/
  export WORKON_HOME=~/.virtualenvs
  export VIRTUALENVWRAPPER_PYTHON=`which python`
  export VIRTUALENVWRAPPER_VIRTUALENV=~/.local/bin/virtualenv
  source ~/.local/bin/virtualenvwrapper.sh
  mkvirtualenv geometry
  # deleting cached packages to force download
  rm -rf ~/.cache/pip/
  pip install -r requirements.txt -I --no-cache-dir
  pip install -e .
else
  export WORKON_HOME=~/.virtualenvs
  export VIRTUALENVWRAPPER_PYTHON=`which python`
  export VIRTUALENVWRAPPER_VIRTUALENV=~/.local/bin/virtualenv
  source ~/.local/bin/virtualenvwrapper.sh
  workon geometry
  # Check if all the package requirements are satisfied
  available_dep_count=$(pip freeze | grep -f requirements.txt | wc -l)
  needed_dep_count=$(cat requirements.txt | wc -l)
  if [ "$available_dep_count" != "$needed_dep_count" ]; then
    echo "Updating dependencies"
    # deleting cached packages to force download
    rm -rf ~/.cache/pip/
    pip install -r requirements.txt -I --no-cache-dir
  fi
fi
