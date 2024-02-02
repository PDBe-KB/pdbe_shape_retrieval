# Shape retrieval for protein surfaces 

## Basic information

This Python package calculates 3D shape descriptors from triangulated molecular surface meshes to analyse protein structure similarity. 

The code is based on pyFM modules and will:

- Process triangulated meshes for protein surfaces
- Calculate 3D Shape descriptors: Wave Kernel Signatures and Heat Kernel Signatures 
- Compute functional maps and refine methods (Zoomout and ICP )
- Compute shape distance matrices 

To install the module ```shape_retrieval``` :
```
git clone https://github.com/PDBe-KB/pdbe_shape-retrieval

cd pdbe_shape-retrieval

python setup.py install

``` 
## Dependencies 

Dependencies can be installed with:

```
pip install -r requirements.txt
```
See  [requirements.txt](https://github.com/PDBe-KB/pdbe_shape-retrieval/blob/main/requirements.txt)


For development: 

**pre-commit usage**

```
pip install pre-commit
pre-commit
pre-commit install
```


## Usage

Follow below steps to install the modules **pdbe_shape-retrieval** 

```
cd pdbe_shape-retrieval/

python setup.py install .

```

To run the modules in command line:

**pdbe_shape-retrieval**: 

```
python pdbe_shape-retrieval/shape_utils/run.py [-h] --input_mesh1 INPUT_FILE_MESH_1 --input_mesh2 INPUT_FILE_MESH_2 -o PATH_TO_OUTPUT_DIR
```
OR 

```
shape_retrieval [-h] --input_mesh1 INPUT_FILE_MESH_1 --input_mesh2 INPUT_FILE_MESH_2 -o PATH_TO_OUTPUT_DIR

```

Required arguments are :

```
--input_mesh1             :  Triangulated mesh for structure 1 (.off)
--input_mesh2             :  Triangulated mesh for structure 2 (.off)    
--output (-o)             :  Output directory
```


Other optional arguments are:

```
--neigvecs      : No. of eigenvalues/eigenvectors to process (>100). A minimum of neigvecs=100 will be automatically set 
--n_ev          : The least number of Laplacian eigenvalues to consider for functional map.
--ndescr        : No. of descriptors to process (WKS/HKS).
--landmarks     : Input indices of landmarks
--step          : Subsample step in order not to use too many descriptors.
--descr         : Type of descriptor to calculate:WKS,HKS,Zernike
--n_cpus        : Number of threads to be used for this calculation.
```

## Versioning

We use [SemVer](https://semver.org) for versioning.

## Authors
* [Grisell Diaz Leines](https://github.com/grisell) - Lead developer
* [Sreenath ](otienoanyango) - Review and productionising

See all contributors [here](https://github.com/PDBe-KB/pisa-analysis/graphs/contributors).

## License

See  [LICENSE](https://github.com/PDBe-KB/pisa-analysis/blob/main/LICENSE)

## Acknowledgements
