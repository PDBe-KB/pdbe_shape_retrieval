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
import importlib.util  
import numpy as np
from io import StringIO

import importlib.util 

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

    mv_command = 'cp ' + fileid + '.* ' + output_dir
    #print(mv_command)
    os.system(mv_command)
    #rm_command = 'rm ' + output_dir + fileid + '.obj'
    #print(rm_command)                                                                                              
    #os.system(rm_command)
    #rm_command = 'rm ' + output_dir + fileid + '.obj.grid'
    #print(rm_command)
    #os.system(rm_command)



def plytoobj(filename,output_dir):
        print(filename[:-4] )
        obj_filename = filename[:-4] + '.obj'
        obj_file = open(obj_filename, 'w')

        with open(filename) as ply_file:
                ply_file_content = ply_file.read().split('\n')[:-1]
                for content in ply_file_content:
                        content_info = content.split()
                        if len(content_info) == 6 and content[0:3] != 'obj':
                                vertex_info = 'v ' + ' '.join(content_info[0:3])
                                obj_file.write(vertex_info + '\n')
                        elif len(content_info) == 7 or len(content_info) == 4:
                                vertex1, vertex2, vertex3 = map(int, content_info[1:4])
                                vertex1, vertex2, vertex3 = vertex1 + 1, vertex2 + 1, vertex3 + 1
                                face_info = 'f ' + str(vertex1) + ' ' + str(vertex2) + ' ' + str(vertex3)
                                obj_file.write(face_info + '\n')

                obj_file.close()
        return obj_filename


def predict_similarity(input_dir,output_dir,module_similarity_path):
       module_path = os.path.join(module_similarity_path,'predict_similarity.py')
       spec = importlib.util.spec_from_file_location("mod", module_path) 
       similarity_module = importlib.util.module_from_spec(spec)
       spec.loader.exec_module(similarity_module)
       
       predict_similarity_command = 'python' + ' ' + module_path+' '+'--input_dir'+' '+input_dir+' '+'--output_dir'+' '+output_dir
       os.system(predict_similarity_command)


       #device_id = 0 
       #cuda = 'true'
       #model_name = 'neural_network'
       #atom_type = 'fullatom'

       #similarity_module.run_predictions(input_dir,output_dir,atom_type,model_name,device_id,cuda)
       
       #atom_type ='mainchain'
       
       #similarity_module.run_predictions(input_dir,output_dir,atom_type,model_name,device_id,cuda)

