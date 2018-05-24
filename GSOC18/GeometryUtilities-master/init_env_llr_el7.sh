#! /usr/bin/env bash

# setup Python 2.7 and ROOT 6
source /opt/exp_soft/llr/root/v6.12.06-el7-gcc49/etc/init.sh

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
  pip install -r requirements_el7.txt -I --no-cache-dir
  pip install -e .
else
  export WORKON_HOME=~/.virtualenvs
  export VIRTUALENVWRAPPER_PYTHON=`which python`
  export VIRTUALENVWRAPPER_VIRTUALENV=~/.local/bin/virtualenv
  source ~/.local/bin/virtualenvwrapper.sh
  workon geometry
  # Check if all the package requirements are satisfied
  available_dep_count=$(pip freeze | grep -f requirements_el7.txt | wc -l)
  needed_dep_count=$(cat requirements_el7.txt | wc -l)
  if [ "$available_dep_count" != "$needed_dep_count" ]; then
    echo "Updating dependencies"
    # deleting cached packages to force download
    rm -rf ~/.cache/pip/
    pip install -r requirements_el7.txt -I --no-cache-dir
  fi
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/llr/info/sartiran/.local/lib

