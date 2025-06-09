import unittest
from unittest.mock import patch, mock_open, call
import os

# Import the functions to be tested (adjust if they're in a different module)
from shape_utils.zernike_descr  import get_inv, plytoobj


class Test3DZernike(unittest.TestCase):
    
    @patch('os.system')
    def test_get_inv(self, mock_system):
        obj_file = 'example.obj'
        fileid = '123'
        map3dz_binary = '/bin/map3dz'
        obj2grid_binary = '/bin/obj2grid'
        output_dir = '/output'

        get_inv(obj_file, fileid, map3dz_binary, obj2grid_binary, output_dir)

        expected_calls = [
            call(f'cp {obj_file} ./123.obj'),
            call(f'/bin/obj2grid -g 64  ./123.obj'),
            call(f'/bin/map3dz 123.obj.grid -c 0.5 '),
            call(f'mv 123.obj.grid.inv 123.inv'),
            call(f'mv 123.* {output_dir}'),
        ]

        mock_system.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(mock_system.call_count, 5)

    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')  # suppress print for clean test output

    def test_plytoobj(self, mock_print, mock_file):
        ply_content = "0.0 0.0 0.0 255 255 255\n" \
                      "0.0 1.0 0.0 255 255 255\n" \
                      "1.0 0.0 0.0 255 255 255\n" \
                      "3 0 1 2\n"
        filename = 'test.ply'

        # mock reading file content
        mock_file().read.return_value = ply_content

        obj_filename = plytoobj(filename, '/output')

        self.assertEqual(obj_filename, 'test.obj')

        expected_writes = [
            call().write('v 0.0 0.0 0.0\n'),
            call().write('v 0.0 1.0 0.0\n'),
            call().write('v 1.0 0.0 0.0\n'),
            call().write('f 1 2 3\n'),
        ]
        mock_file.assert_has_calls(expected_writes, any_order=False)


