# Shape retrieval for protein surfaces 

## Basic information

This Python package calculates 3D shape descriptors for triangulated molecular surface meshes to analyse protein structure similarity. 

The code is based on [pyFM](https://github.com/RobinMagnet/pyFM) modules and [3D-Surfer 2.0](https://kiharalab.org/3d-surfer/) code and will:

- Process triangulated meshes for protein surfaces
- Calculate 3D Shape descriptors for two protein structures: Wave Kernel Signatures (WKS), Heat Kernel Signatures (HKS) and 3D Zernike descriptors (3DZD)
- Compute functional maps and refine methods (Zoomout and ICP )
- Compute similarity scores
- Provides analysis tools to compute a score square matrix and perform agglomerative clustering 

To install the module ```shape_retrieval``` :
```
git clone https://github.com/PDBe-KB/pdbe_shape-retrieval

cd pdbe_shape-retrieval

python setup.py install

``` 
## Dependencies 

This package requires the installation of [pyFM](https://github.com/RobinMagnet/pyFM) module:

```
pip install pyfmaps
```

Other dependencies can be installed with:

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

Follow the steps below to install the modules **pdbe_shape-retrieval** 

```
cd pdbe_shape-retrieval/

python setup.py install .

```

To run the modules in the command line:

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
--entry_ids               :  Entry IDs for protein structures 
--output (-o)             :  Output directory
```


Other optional arguments are:

For pre-processing and fixing faulty meshes:
```
--fix_meshes    : Fix and clean meshes using pymeshfix to obtain well-conditioned meshes for the calculation of shape descriptors.
--collapse_vertices : Collapse the number of vertices in the mesh to reduce the resolution using decimation quadric edge collapse. This option must be used with --fix_meshes
--resolution : Factor to collapse No. of vertices, e.g 0.5 will reduce the vertices to ~half. The default is 0.5. This option should be used with --collapse_vertices.
--reconstruct_mesh: Reconstruct the mesh to obtain a new well-conditioned mesh using VCG surface reconstruction. This option must be used with --fix_meshes.
```
Select the shape descriptor you wish to compute:
```
--descr : Type of descriptor to calculate: WKS,HKS,3DZD. The default is WKS.
```

Options for the calculation of spectral descriptors:
```
--neigvecs      : No. of eigenvalues/eigenvectors to process (>100). A minimum of neigvecs=100 will be used by default (recommended) 
--n_ev          : The least number of Laplacian eigenvalues to consider for the functional map.
--ndescr        : No. of descriptors to process (WKS/HKS).
--landmarks     : Input indices of landmarks
--step          : Subsample step to avoid using too many descriptors.
--descr         : Type of descriptor to calculate: WKS,HKS,Zernike
--n_cpus        : Number of threads to be used for the calculation of functional maps.
--refine        : Use refining method for calculation of fuctional maps: icp,zoomout
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
