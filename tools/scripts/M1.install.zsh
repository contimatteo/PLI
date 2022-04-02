#

pip install -U pip
pip install wheel

#

export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1   

pip install cython
pip install --no-binary=h5py h5py
pip install grpcio

#

git clone https://github.com/pandas-dev/pandas.git
cd pandas
python setup.py install
cd ..
rm -rf pandas

# 

pip install -r ./tools/requirements/M1.txt 
