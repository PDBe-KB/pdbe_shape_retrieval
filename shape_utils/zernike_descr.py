# Copyright (C) 2021 Tunde Aderinwale, Daisuke Kihara, and Purdue University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# convert pdb file to triangulation data and do 3dzd calculation

import os
import shutil
import importlib.util  
import importlib.util 
import subprocess
import os
import logging


def get_inv(obj_file,fileid,map3dz_binary, obj2grid_binary,output_dir):
        """
        Generates Zernike moments in .inv file from a 3D OBJ mesh file by converting it to a grid 
        and applying 3D-Surface binary map3dz (Kihara's lab).

        Args:

        obj_file (str): Path to the input mesh OBJ file.         
        fileid (str): Identifier used for naming output files.
        map3dz_binary (str): Path to the `map3dz` binary executable.
        obj2grid_binary (str): Path to the `obj2grid` binary executable.
        output_dir (str): Path to directory where output files will be stored.

        Raises:
        FileNotFoundError:
            If the input OBJ file or binaries do not exist.
        subprocess.CalledProcessError:
            If `obj2grid` or `map3dz` fails.
        OSError:
            If renaming or deleting temporary files fails.
        """

        obj_output = os.path.join(output_dir,"{}.obj".format(fileid))
        

        # --- 1. COPY OBJ FILE -------------------------------------------------
        if not os.path.abspath(obj_file) == os.path.abspath(obj_output):
                shutil.copy(obj_file, obj_output)
        else:
                logging.info("Skipping copy: source and destination are identical.")
                print("Skipping copy: source and destination are identical.")

        # --- 2. RUN obj2grid --------------------------------------------------
        subprocess.run(
            [obj2grid_binary, "-g", "64", str(obj_output)],
           check=True
        )
        
        # generate 3dzd
        # --- 3. RUN map3dz ----------------------------------------------------
        grid_file = os.path.join(output_dir,"{}.obj.grid".format(fileid))
        
        subprocess.run(
            [map3dz_binary, f"{grid_file}", "-c", "0.5"],
            check=True
        )

        # --- 4. RENAME .inv FILE ----------------------------------------------
        
        inv_file = grid_file+".inv"
        final_inv = os.path.join(output_dir,"{}.inv".format(fileid))
        os.rename(inv_file, final_inv)

        # Delete .obj and .grid files from output folder
        if not os.path.abspath(obj_file) == os.path.abspath(obj_output):
                os.remove(obj_output)
        os.remove(grid_file)                







