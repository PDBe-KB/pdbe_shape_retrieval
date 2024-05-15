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
import glob
import sys
import numpy as np
from io import StringIO


def get_inv(obj_file,fileid,map3dz_binary, obj2grid_binary,output_dir):
	
    # generate 3dzd

    cp_command = 'cp ' + obj_file + ' ./' + fileid + '.obj'
    #print(cp_command)
    os.system(cp_command)

    grid_command = obj2grid_binary+' '+'-g 64  ./' + fileid + '.obj'
    #print(grid_command)
    os.system(grid_command)
    
    inv_command =  map3dz_binary +' '+ fileid + '.obj.grid -c 0.5 '#--save-moments'
    #print(inv_command)
    os.system(inv_command)
    
    mv_command = 'mv ' + fileid + '.obj.grid.inv ' + fileid + '.inv'
    #print(mv_command)
    os.system(mv_command)

    mv_command = 'mv ' + fileid + '.* ' + output_dir
    #print(mv_command)
    os.system(mv_command)
    rm_command = 'rm ' + output_dir + fileid + '.obj'
    #print(rm_command)                                                                                              
    os.system(rm_command)
    rm_command = 'rm ' + output_dir + fileid + '.obj.grid'
    #print(rm_command)
    os.system(rm_command)



