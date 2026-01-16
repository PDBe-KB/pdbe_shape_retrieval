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

This package requires the installation of [pyFM](https://github.com/RobinMagnet/pyFM) module for the calculation of spectral descriptors:

```
pip install pyfmaps
```

For the calculation of Zernike Descriptors, binaries `obj2grid` and `map2zernike` from 3D-Surfer and obj2grid codes need to be provided. 
The binaries are available [here](https://github.com/PDBe-KB/pdbe_shape-retrieval/blob/main/bin)

To make your life easier when running the process, it is better to set two path environment variables for 3D-Surfer:

An environment variable to the `obj2grid` binary:

```
export PATH="$PATH:your_path_to_obj2grid/obj2grid"
```

A path to the `map2zernike binary` of 3D-Surfer :

```
export PISA_SETUP_DIR="/your_path_to_3DSurfer/bin/"
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
python pdbe_shape-retrieval/shape_utils/run.py [-h] --input_mesh1 INPUT_FILE_MESH_1 --input_mesh2 INPUT_FILE_MESH_2  --entry_ids ENTRY_ID_1 ENTRY_ID_2  -o PATH_TO_OUTPUT_DIR
```
OR 

```
shape_retrieval [-h] --input_mesh1 INPUT_FILE_MESH_1 --input_mesh2 INPUT_FILE_MESH_2 --entry_ids ENTRY_ID_1 ENTRY_ID_2 -o PATH_TO_OUTPUT_DIR

```

Required arguments are :

```
--input_mesh1             :  Triangulated mesh for structure 1 (.off)
--input_mesh2             :  Triangulated mesh for structure 2 (.off)
--entry_ids               :  Entry IDs for protein structures 
--output (-o)             :  Output directory
```


Other optional arguments:

For pre-processing and fixing faulty meshes:
```
--fix_meshes    : Fix and clean meshes using pymeshfix to obtain well-conditioned meshes for the calculation of shape descriptors.
--collapse_vertices : Collapse the number of vertices in the mesh to reduce the resolution using decimation quadric edge collapse. This option must be used with --fix_meshes
--resolution : Factor to collapse No. of vertices, e.g 0.5 will reduce the vertices to ~half. The default is 0.5. This option should be used with --collapse_vertices.
--reconstruct_mesh: Reconstruct the mesh to obtain a new well-conditioned mesh using VCG surface reconstruction. This option must be used with --fix_meshes.
```
To select the shape descriptor:
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
--n_cpus        : Number of threads used to calculate functional maps.
--refine        : Use refining method for calculation of functional maps: icp,zoomout

```
Options for the calculation of Zernike descriptors:

```
--map2zernike_binary : Path to map2zernike binary
--obj2grid_binary : 'Path to obj2grid_binary'
```
Other options:

```
--min_dist_mesh : Calculate the minimum distance between two meshes
--no_shape_retrieval: Switch off the calculation of shape descriptors
```

## Expected output files for Spectral descriptors

The process will output CSV files with the WKS/HKS descriptors for the two input meshes:

*DESCR_TYPE_descr_ENTRY_ID_1.csv*
*DESCR_TYPE_descr_ENTRY_ID_2.csv*

where DESCR_TYPE is the selected descriptor (WKS) and ENTRY_ID is the entry id of the input structure. 

A vector of N descriptors is given for each vertex of the mesh, and therefore the No. of rows of the file is the no. of vertices of the mesh:

In the following example N=5 and the output file would look like this:
```
wks_descriptors[5]
0.18651766218142352,0.18146979432734137,0.17754398826544535,0.1750789523051364,0.1746571180966082
0.4586452611353971,0.6323283658831136,0.8661039157025516,1.1267042526529782,1.3487451570690734
1.4914506660788938,1.5658528547661157,1.6007092706487118,1.6165954563778082,1.623964369261013
1.6274787893755465,1.6291230475034042,1.6297139199237025,1.6295464581278336,1.628654203946087
.
.
(No. of rows = No. of vertices in mesh file)

```

The process will output a CSV file with the correspondence matrix or functional map (FM):

*ENTRY_ID_1_ENTRY_ID_2_FM.csv*

where ENTRY_ID_* is the entry ID for each input structure.

The csv file contains NxN rows and columns with values of the transformation coefficients c_ij. N is the number of terms or Laplace-Beltrami functions used in the expansion in fuctional space. 

For example if N=4 the output file would look like this:
```
-1.0,5.363410999415223e-05,4.484982613902667e-06,8.651562620111304e-06,-7.754014997702692e-06
9.920763930058441e-06,7.5488644078142375e-06,-3.4094219364651095e-06,-1.408423031071398e-05
-4.28569154629773e-06,-3.394752296984109e-06,3.821338048287914e-06,1.177611477637059e-05
-1.59927066983662e-06,-7.931408043822854e-06,9.907381306458515e-06,1.1689248196168793e-05
```
The process will output a CSV file with the point-to-point map from mesh2 to mesh1:

*ENTRY_ID_1_ENTRY_ID_2_p2p21.csv*

In this output file the No. of rows N corresponds to the no. of vertices in Mesh 2 and each row displays the corresponding vertex index of Mesh 1. 

```
8
9
10
10
1
1
.
.
(No. of rows in output file is the No. of vertices in Mesh 2)
```
## Expected output files for Zernike descriptors

## Versioning

We use [SemVer](https://semver.org) for versioning.

## Authors
* [Grisell Diaz Leines](https://github.com/grisell) - Lead developer
* [Sreenath ](otienoanyango) - Review and productionising

See all contributors [here](https://github.com/PDBe-KB/pisa-analysis/graphs/contributors).

## License

See  [LICENSE](https://github.com/PDBe-KB/pisa-analysis/blob/main/LICENSE)

## Acknowledgements
