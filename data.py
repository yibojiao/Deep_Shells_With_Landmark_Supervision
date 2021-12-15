import os
import numpy as np
import torch
from shape_utils import *
from typing import List
from scipy.spatial import distance_matrix
from plyfile import PlyData

# pp to array
def pp_to_array(pp):
    file = open(pp, 'r')
    file.readline()
    file.readline()
    verts = []
    while True:
        l = file.readline().strip().split(' ')
        if l[0] == '</PickedPoints>':
            break
        v = [l[1].split('"')[1], l[2].split('"')[1], l[3].split('"')[1]]
        verts.append(v)
    return verts

def euclidean_dist_squared(lnds, verts):
    return distance_matrix(lnds, verts)

# find the nearest point of the landmark on the shape
def find_min_dis_vert(lnds, verts):
    dist = np.sqrt(euclidean_dist_squared(lnds, verts))
    dist[np.isnan(dist)] = np.inf
    return np.argmin(dist, axis=1)

def input_to_batch(mat_dict):
    dict_out = dict()

    for attr in ["vert", "triv", "evecs", "evals", "SHOT"]:
        if mat_dict[attr][0].dtype.kind in np.typecodes["AllInteger"]:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.int32)
        else:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.float32)

    for attr in ["A"]:
        dict_out[attr] = np.asarray(mat_dict[attr][0].diagonal(), dtype=np.float32)

    return dict_out

def read_ply_file(ply_file):
    ply_file = '{}.ply'.format(ply_file)
    ply_data = PlyData.read(ply_file)
    verts_data = ply_data['vertex'].data
    verts = np.zeros((verts_data.size, 3))
    verts[:, 0] = verts_data['x']
    verts[:, 1] = verts_data['y']
    verts[:, 2] = verts_data['z']  
    return verts

def input_to_batch_with_lnd(mat_dict, name):
    dict_out = dict()

    for attr in ["vert", "triv", "evecs", "evals", "SHOT"]:
        if mat_dict[attr][0].dtype.kind in np.typecodes["AllInteger"]:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.int32)
        else:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.float32)

    for attr in ["A"]:
        dict_out[attr] = np.asarray(mat_dict[attr][0].diagonal(), dtype=np.float32)
    
    # save landmark vertext index on shape
    #lnd = np.array(pp_to_array('../CAESAR_train/lnd/{}.pp'.format(name[:-4]))).astype('float')
    lnd = np.array(pp_to_array('data/lnd/{}.pp'.format(name[:-4]))).astype('float')
    dict_out["lnd_idx"] = find_min_dis_vert(lnd, read_ply_file('data/ply/{}'.format(name[:-4])))
    print(dict_out["lnd_idx"])
    return dict_out


def batch_to_shape(batch):
    shape = Shape(batch["vert"].squeeze().to(device), batch["triv"].squeeze().to(device, torch.long) - 1)

    for attr in ["evecs", "evals", "SHOT", "A"]:
        setattr(shape, attr, batch[attr].squeeze().to(device))
        
    shape.lnd = batch["lnd_idx"][0]
    shape.compute_xi_()

    return shape


class ShapeDatasetOnePair(torch.utils.data.Dataset):
    def __init__(self, file_name_1, file_name_2=None):

        load_data = scipy.io.loadmat(file_name_1)

        self.data_x = input_to_batch(load_data["X"][0])

        if file_name_2 is None:
            self.data_y = input_to_batch(load_data["Y"][0])
            print("Loaded file ", file_name_1, "")
        else:
            load_data = scipy.io.loadmat(file_name_2)
            self.data_y = input_to_batch(load_data["X"][0])
            print("Loaded files ", file_name_1, " and ", file_name_2)

    def _get_index(self, i):
        return i

    def __getitem__(self, index):
        data_curr = dict()
        if index == 0:
            data_curr["X"] = self.data_x
            data_curr["Y"] = self.data_y
        else:
            data_curr["X"] = self.data_y
            data_curr["Y"] = self.data_x
        return data_curr

    def __len__(self):
        return 2


class ShapeDatasetCombine(torch.utils.data.Dataset):
    def __init__(self, file_fct, num_shapes):
        self.file_fct = file_fct
        self.num_shapes = num_shapes
        self.num_pairs = num_shapes ** 2

        self.data = []

        self._init_data()

    def _init_data(self):
        for i in range(self.num_shapes):
            file_name, shape_name = self.file_fct(self._get_index(i))
            load_data = scipy.io.loadmat(file_name)

            data_curr = input_to_batch_with_lnd(load_data["X"][0], shape_name)

            self.data.append(data_curr)

            print("Loaded file ", file_name, "")

    def _get_index(self, i):
        return i

    def __getitem__(self, index):
        i1 = int(index / self.num_shapes)
        i2 = int(index % self.num_shapes)
        data_curr = dict()
        data_curr["X"] = self.data[i1]
        data_curr["Y"] = self.data[i2]
        return data_curr

    def __len__(self):
        return self.num_pairs


class ShapeDatasetCombineMulti(ShapeDatasetCombine):
    def __init__(self, datasets: List[ShapeDatasetCombine]):
        self.datasets = datasets
        num_shapes = sum([d.num_shapes for d in datasets])
        super().__init__(None, num_shapes)

    def _init_data(self):
        for d in self.datasets:
            self.data += d.data


def get_faustremeshed_file(i):
    #folder_path = "../CAESAR_train/processed"
    folder_path = "data/processed"
    assert folder_path != "", "Specify the location of FAUST remeshed"
    faust_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    faust_files.sort()
    return os.path.join(folder_path, faust_files[i]), faust_files[i]


class Faustremeshed_train(ShapeDatasetCombine):
    def __init__(self):
        super().__init__(get_faustremeshed_file, 1)
        print("loaded FAUST_remeshed with " + str(self.num_pairs) + " pairs")

