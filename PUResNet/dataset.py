# _date_:2021/8/27 12:18
from torch.utils.data import Dataset
import h5py
from random import shuffle, choice, sample
import numpy as np
from scipy import ndimage
import tfbio
import tfbio.data
from skimage.draw import ellipsoid
import glob
from os.path import join
from pybel import readfile
import os
from tfbio.data import Featurizer
from skimage.measure import label

# hdf
# ('coords', prot_coords),
# ('features', prot_features),
# ('pocket_coords', pocket_coords),
# ('pocket_features', pocket_features),
# ('centroid', centroid))

DATA_ROOT = ''

class BaseTrainSet(Dataset):
    def __init__(self):
        super(BaseTrainSet, self).__init__()
        self.max_dist = 35
        self.scale = 0.5
        self.footprint = None
        self.max_translation = 5
        self.transform = True
        hdf_path = join(DATA_ROOT, 'scpdb_dataset.hdf')
        self.data_handle = h5py.File(hdf_path, mode='r')
        footprint = ellipsoid(2, 2, 2)
        self.footprint = footprint.reshape((1, *footprint.shape, 1))
        pdbids = list(self.data_handle.keys())
        self.x_channels = self.data_handle[pdbids[0]]['features'].shape[1]
        self.y_channels = self.data_handle[pdbids[0]]['pocket_features'].shape[1]


class TrainscPDB(BaseTrainSet):
    def __init__(self, subset, one_channel=False):
        super(TrainscPDB, self).__init__()
        if subset == 'train':

            with open(join(DATA_ROOT, 'ten_folds/train_ids_fold0')) as f:
                lines = f.readlines()
            self.pdbids = [line.strip() for line in lines]
        elif subset == 'validation':
            self.transform = False
            with open(join(DATA_ROOT, 'ten_folds/test_ids_fold0')) as f:
                lines = f.readlines()
            self.pdbids = [line.strip() for line in lines]

        self.one_channel = one_channel
        print('dataset_len=', len(self.pdbids))

    def __getitem__(self, index):  # sample_generator
        if index == 0:
            shuffle(self.pdbids)
        pdbid = self.pdbids[index]

        if self.transform:
            rot = choice(range(24))
            tr = self.max_translation * np.random.rand(1, 3)
        else:
            rot = 0
            tr = (0, 0, 0)
        r, p = self.prepare_complex(pdbid, rotation=rot, translation=tr)  # (1, 36, 36, 36, 18) (1, 36, 36, 36, 1)
        r, p = np.squeeze(r, 0), np.squeeze(p, 0)  # (36, 36, 36, 18) (36, 36, 36, 1)
        r, p = r.transpose((3, 0, 1, 2)), p.transpose((3, 0, 1, 2))
        return r, p

    def __len__(self):
        return len(self.pdbids)

    def prepare_complex(self, pdbid, rotation=0, translation=(0, 0, 0), vmin=0, vmax=1):
        """Prepare complex with given pdbid.

        Parameters
        ----------
        pdbid: str
            ID of a complex to prepare
        rotation: int or np.ndarray (shape (3, 3)), optional (default=0)
            Rotation to apply. It can be either rotation matrix or ID of
            rotatation defined in `tfbio.data` (0-23)
        translation: tuple of 3 floats, optional (default=(0, 0, 0))
            Translation to apply
        vmin, vmax: floats, optional (default 0 and 1)
            Clip values generated for pocket to this range

        Returns
        -------
        rec_grid: np.ndarray
            Grid representing protein
        pocket_dens: np.ndarray
            Grid representing pocket
        """

        resolution = 1. / self.scale
        structure = self.data_handle[pdbid]
        rec_coords = tfbio.data.rotate(structure['coords'][:], rotation)
        rec_coords += translation
        if self.one_channel:
            features = np.ones((len(rec_coords), 1))
        else:
            features = structure['features'][:]
        rec_grid = tfbio.data.make_grid(rec_coords, features,
                                        max_dist=self.max_dist,
                                        grid_resolution=resolution)
        # print('protien:', rec_grid.shape) # (1, 36, 36, 36, 18)

        pocket_coords = tfbio.data.rotate(structure['pocket_coords'][:], rotation)
        pocket_coords += translation

        # print('=====', pocket_coords.shape, structure['pocket_features'][:].shape) # (point_num, 3)(point_num, 1)
        pocket_dens = tfbio.data.make_grid(pocket_coords,
                                           structure['pocket_features'][:],
                                           max_dist=self.max_dist)
        # print('456grid', pocket_dens.shape)  # (1, 71, 71, 71, 1)
        margin = ndimage.maximum_filter(pocket_dens, footprint=self.footprint)
        pocket_dens += margin
        pocket_dens = pocket_dens.clip(vmin, vmax)

        # print('456minmax', pocket_dens.shape) # (1, 71, 71, 71, 1)

        # print('x_channel, y_channel=', self.x_channels, self.y_channels)

        zoom = rec_grid.shape[1] / pocket_dens.shape[1]
        pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i], zoom) for i in range(self.y_channels)], -1)
        # print('456---:', pocket_dens.shape)  # (36, 36, 36, 1)
        pocket_dens = np.expand_dims(pocket_dens, 0)  # (1, 36, 36, 36, 1)
        pocket_dens = pocket_dens.clip(vmin, vmax)
        # print('456===:', pocket_dens.shape)
        return rec_grid, pocket_dens


class BaseTestSet(Dataset):
    def __init__(self, scale=0.5, max_dist=35):
        super(BaseTestSet, self).__init__()
        self.scale = scale
        self.max_dist = max_dist
        self.featurizer = Featurizer(save_molecule_codes=False)
        footprint = ellipsoid(2, 2, 2)
        self.footprint = footprint.reshape((1, *footprint.shape, 1))
        self.y_channels = 1

        self.latent_data_handle = None
        self.resolution = 1. / self.scale
        self.centroid = None

    def get_prior_pocket(self, pdbid):
        ''' latent '''
        structure_latent = self.latent_data_handle[pdbid]
        # self.origin, self.step = structure_latent['origin'], structure_latent['step']
        latent_coords = np.array(structure_latent['index'])
        coords = latent_coords * 2 - 35
        features = np.ones((len(coords), 1))
        latent_grid = tfbio.data.make_grid(coords, features,
                                           max_dist=self.max_dist,
                                           grid_resolution=self.resolution)  # (1, 36, 36, 36, 1)
        return latent_grid

    def get_mol(self, path):
        # print('======', path)
        pdb_name = path.split('/')[-2]
        suffix = path.split('.')[-1]
        mol = next(readfile(suffix, path))
        coords = np.array([a.coords for a in mol.atoms])
        # print(len(coords), coords[:3])
        centroid = coords.mean(axis=0)
        coords -= centroid
        return centroid, mol

    def density_form_mol(self, mol, one_channel):
        prot_coords, prot_features = self.featurizer.get_features(mol)
        # print(len(prot_coords), prot_coords[:3])
        self.centroid = prot_coords.mean(axis=0)
        prot_coords -= self.centroid
        resolution = 1. / self.scale
        if one_channel:
            features = np.ones((len(prot_coords), 1))
        else:
            features = prot_features
        # print(np.ones((len(prot_coords), 1)).shape, prot_features.shape) #(2232, 1) (2232, 18)
        x = tfbio.data.make_grid(prot_coords, features,
                                 max_dist=self.max_dist,
                                 grid_resolution=resolution)  # (1, 36, 36, 36, 18)
        x = np.squeeze(x, 0)
        x = x.transpose((3, 0, 1, 2))  # (18, 36, 36, 36)

        return x

    def density_form_mol_01(self, mol):
        prot_coords, prot_features = self._get_binary_features(mol)
        # print(len(prot_coords), prot_coords[:3])
        centroid = prot_coords.mean(axis=0)
        prot_coords -= centroid
        resolution = 1. / self.scale
        x = tfbio.data.make_grid(prot_coords, prot_features,
                                 max_dist=self.max_dist,
                                 grid_resolution=resolution)  # (1, 36, 36, 36, 1)
        x = np.squeeze(x)  # (36, 36, 36)
        # x = x.transpose((3, 0, 1, 2))  # (1, 36, 36, 36)
        origin = (centroid - self.max_dist)
        step = np.array([1.0 / self.scale] * 3)
        return x, origin, step

    def get_label_grids(self, cavity_paths, protein_centroid):
        # cavity_paths = glob.glob(join(dir_path, self.cavity_keyword))
        # print(dir_path, self.suffix)
        # print('len_cavity=', len(cavitys), dir_path)
        label_grids = np.zeros(shape=(1, 36, 36, 36, 1))
        pocket_number = 0
        # print(cavity_paths[0])
        cavity_suffix = cavity_paths[0].split('.')[-1]
        # print(cavity_paths)
        for n, cavity_path in enumerate(cavity_paths, start=1):
            if not os.path.exists(cavity_path):
                continue
            mol = next(readfile(cavity_suffix, cavity_path))
            pocket_coords, pocket_features = self._get_binary_features(mol)
            pocket_coords -= protein_centroid
            resolution = 1. / self.scale
            x = tfbio.data.make_grid(pocket_coords, pocket_features,
                                     max_dist=self.max_dist,
                                     grid_resolution=resolution)  # (1, 36, 36, 36, 1)
            x = np.where(x > 0, 1, 0)
            if not (x == 1).any():  # pocket超出边界
                # print('###################################################')
                continue
            pocket_number += 1
            x = x * pocket_number
            label_grids += x
            label_grids = np.where(label_grids > pocket_number, pocket_number, label_grids).astype(int) # 含0，1，2，3...表示的矩阵

        old_pocket_number = pocket_number
        for p in range(1, pocket_number + 1):
            while not (p in label_grids) and p <= np.max(label_grids):
                label_grids = np.where(label_grids > p, label_grids - 1, label_grids)
                pocket_number -= 1
        # print('cavity_path=', cavity_paths, 'now=', pocket_number, 'old=', old_pocket_number)
        return label_grids.astype(int), pocket_number

    def get_label_01(self, cavity_paths, protein_centroid, vmin=0, vmax=1, size=36):
        # cavity_paths = glob.glob(join(dir_path, self.cavity_suffix))
        # print(dir_path, self.suffix)
        # print('len_cavity=', len(cavitys))
        cavity_suffix = cavity_paths[0].split('.')[-1]
        label_grids = None
        pocket_coords, pocket_features = None, None
        for n, cavity_path in enumerate(cavity_paths, start=1):
            mol = next(readfile(cavity_suffix, cavity_path))
            tmp_pocket_coords, tmp_pocket_features = self._get_binary_features(mol)  # (point_num, 3)(point_num, 1)
            tmp_pocket_coords -= protein_centroid

            if n == 1:
                pocket_coords = tmp_pocket_coords
                pocket_features = tmp_pocket_features
            else:
                pocket_coords = np.concatenate((pocket_coords, tmp_pocket_coords))
                pocket_features = np.concatenate((pocket_features, tmp_pocket_features))

        # resolution = 1. / self.scale
        # print('=====', pocket_coords.shape, structure['pocket_features'][:].shape) # (point_num, 3)(point_num, 1)
        pocket_dens = tfbio.data.make_grid(pocket_coords, pocket_features, max_dist=self.max_dist)
        # print('456grid', pocket_dens.shape)  # (1, 71, 71, 71, 1)
        margin = ndimage.maximum_filter(pocket_dens, footprint=self.footprint)
        pocket_dens += margin
        pocket_dens = pocket_dens.clip(vmin, vmax)
        # print('456minmax', pocket_dens.shape) # (1, 71, 71, 71, 1)
        # print('x_channel, y_channel=', self.x_channels, self.y_channels)
        # zoom = rec_grid.shape[1] / pocket_dens.shape[1]
        zoom = size / pocket_dens.shape[1]
        pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i], zoom) for i in range(self.y_channels)], -1)
        # print('456---:', pocket_dens.shape)  # (36, 36, 36, 1)
        # pocket_dens = np.expand_dims(pocket_dens, 0)  # (1, 36, 36, 36, 1)
        pocket_dens = pocket_dens.transpose((3, 0, 1, 2))
        return pocket_dens

    def _get_binary_features(self, mol):
        coords = []
        for a in mol.atoms:
            coords.append(a.coords)
        coords = np.array(coords)
        features = np.ones((len(coords), 1))
        return coords, features

    def get_mask(self, pdbid):
        structure_latent = self.latent_data_handle[pdbid]
        self.origin, self.step = structure_latent['origin'], structure_latent['step']
        latent_coords = np.array(structure_latent['index'])
        coords = latent_coords * 2 - 35
        features = np.ones((len(coords), 1))
        latent_grid = tfbio.data.make_grid(coords, features,
                                           max_dist=self.max_dist,
                                           grid_resolution=self.resolution)  # (1, 36, 36, 36, 1)
        latent_grid = np.squeeze(latent_grid, 0)  # (36, 36, 36, 1)
        latent_grid = latent_grid.transpose((3, 0, 1, 2))  # (1, 36, 36, 36)
        return latent_grid



class TestPDBbind(BaseTestSet):
    # '''
    ''' ##### icme camera-ready '''
    def __init__(self, one_channel, mask, is_dca):
        super(TestPDBbind, self).__init__()
        self.is_dca = is_dca
        self.one_channel = one_channel
        self.mask = mask
        if mask:
            mask_path = join(DATA_ROOT, 'pdbbind_latent_pockets_v2.hdf')
            self.latent_data_handle = h5py.File(mask_path, mode='r')

        path = join(DATA_ROOT, 'PDBbind_v2020_refined/refined-set-no-solvent/*/*_protein.pdb')
        # black_list: Too large to load
        black_list = ['3t0b_protein.pdb', '3t09_protein.pdb', '3mv0_protein.pdb', '3dyo_protein.pdb', '3vd4_protein.pdb',
                      '3vdb_protein.pdb', '3f34_protein.pdb', '3i3b_protein.pdb', '3k1j_protein.pdb', '3f37_protein.pdb',
                      '3f33_protein.pdb', '3t08_protein.pdb', '3vd9_protein.pdb', '3t0d_protein.pdb', '3muz_protein.pdb',
                      '3t2q_protein.pdb', '2f2h_protein.pdb', '1px4_protein.pdb']
        # repeat_list: appear in training set
        repeat_list = open(join(DATA_ROOT, 'PDBbind_v2020_refined/repeat_list_1405.txt')).readlines()
        repeat_list = [name.strip() for name in repeat_list]
        total_list = black_list + repeat_list

        remove_list = open(join(DATA_ROOT, 'PDBbind_v2020_refined/remove_list.txt')).readlines()
        remove_list = [name.strip() for name in remove_list]
        total_list += remove_list

        protein_paths = glob.glob(path)
        protein_paths.sort()

        # protein_paths = [p for p in protein_paths if os.path.basename(p) not in total_list]
        # self.protein_paths = []
        # for path in protein_paths:
        #     if len(glob.glob(join(os.path.dirname(path), '*ALL.mol2'))) > 0:
        #         self.protein_paths.append(path)
        # self.cavity_paths = [[path.replace(os.path.basename(path), 'CAVITY_N1_ALL.mol2')] for path in self.protein_paths]

        self.protein_paths = [p for p in protein_paths if os.path.basename(p) not in total_list]
        self.cavity_paths = [[path.replace('protein', 'pocket')] for path in self.protein_paths]
        # self.cavity_paths = [[path.replace(os.path.basename(path), 'CAVITY_N1_ALL.mol2')] for path in self.protein_paths]
        self.ligand_paths = [[path.replace('protein.pdb', 'ligand.mol2')] for path in self.protein_paths]

        self.protein_format = 'pdb'
        self.protein_paths.sort()
        print('all_data=', len(self.protein_paths), len(self.cavity_paths), len(self.ligand_paths))

    def __getitem__(self, index):
        path = self.protein_paths[index]
        # print(path)
        centroid, mol = self.get_mol(path)
        protein_x = self.density_form_mol(mol, self.one_channel)  # # (18, 36, 36, 36)

        # truth_labels, num_cavity = self.get_pocket(self.cavity_paths[index], centroid)  # (1, 36, 36, 36, 1)
        # print(self.protein_paths[index], self.cavity_paths[index])

        # dca_truth_labels, dca_num_cavity = self.get_label_grids(self.ligand_paths[index], centroid)  # (1, 36, 36, 36, 1)
        # truth_labels, num_cavity = self.get_label_grids(self.cavity_paths[index], centroid)  # (1, 36, 36, 36, 1)
        # if self.is_dca:
        #     truth_labels, num_cavity = dca_truth_labels, dca_num_cavity

        if self.is_dca:
            truth_labels, num_cavity = self.get_label_grids(self.ligand_paths[index], centroid)  # (1, 36, 36, 36, 1)
            label_num = len(self.ligand_paths[index])
        else:
            truth_labels, num_cavity = self.get_label_grids(self.cavity_paths[index], centroid)  # (1, 36, 36, 36, 1)
            label_num = len(self.cavity_paths[index])

        # print('num_cavity=', num_cavity)
        truth_labels = np.squeeze(truth_labels)  # # (36, 36, 36)
        # label_01 = self.get_label_01(os.path.dirname(path), centroid, self.data_format)  # (1, 36, 36, 36)
        if self.mask:
            # TODO add mask
            pdbid = os.path.basename(path).split('.')[0]
            latent_grid = self.get_mask(pdbid)  # (1, 36, 36, 36)
            return protein_x, truth_labels, latent_grid

        # print(protein_x.shape, truth_labels.shape, self.centroid)
        return protein_x, truth_labels, self.centroid, label_num
        # return 1, 2, 3

    def __len__(self):
        return len(self.protein_paths)

# TestPDBbind(False, False)

class Test_coach420_holo4k(BaseTestSet):
    def __init__(self, set, is_dca):
        super(Test_coach420_holo4k, self).__init__()
        # set = 'coach420' or 'holo4k'
        protein_root = join(DATA_ROOT, '{}/protein/'.format(set))
        cavity_root = join(DATA_ROOT, '{}/cavity/'.format(set))
        ligand_root = None
        if set == 'coach420':
            ligand_root = join(DATA_ROOT, '{}/ligand_T2_cavity/'.format(set))
        elif set == 'holo4k':
            ligand_root = join(DATA_ROOT, '{}/ligand/'.format(set))

        exist_id = os.listdir(cavity_root)
        exist_id.sort()
        self.protein_paths = [join(protein_root, '{}.pdb'.format(id_)) for id_ in exist_id]

        self.cavity_paths = []
        self.ligand_paths = []
        for id_ in exist_id:
            # print(id_)
            tmp_cavity_paths = glob.glob(join(cavity_root, id_, '*', 'CAVITY*'))
            if set == 'coach420':
                tmp_ligand_paths = glob.glob(join(ligand_root, id_, 'ligand*'))
            elif set == 'holo4k':
                tmp_ligand_paths = glob.glob(join(cavity_root, id_, '*', 'ligand*'))
            # print(tmp_cavity_paths)
            self.cavity_paths.append(tmp_cavity_paths)
            self.ligand_paths.append(tmp_ligand_paths)
        print(len(self.protein_paths), len(self.cavity_paths), len(self.ligand_paths))

        self.one_channel = False

        if is_dca:
            self.label_paths = self.ligand_paths
        else:
            self.label_paths = self.cavity_paths

    def __getitem__(self, index):
        protein_path = self.protein_paths[index]
        centroid, mol = self.get_mol(protein_path)
        protein_x = self.density_form_mol(mol, self.one_channel)  # # (18, 36, 36, 36)
        truth_labels, num_cavity = self.get_label_grids(self.label_paths[index], centroid)  # (1, 36, 36, 36, 1)
        truth_labels = np.squeeze(truth_labels)  # # (36, 36, 36)
        # return protein_x, truth_labels, centroid

        # str_ligand_paths = '+'.join(self.cavity_paths[index])
        # print(str_ligand_paths)

        # return protein_x, truth_labels, '{}={}'.format(protein_path, str_ligand_paths)
        return protein_x, truth_labels, centroid, len(self.cavity_paths[index])
        # return protein_x, truth_labels, centroid, '{}={}'.format(protein_path, str_ligand_paths)
    def __len__(self):
        return len(self.protein_paths)

class Test_sc6k(BaseTestSet):
    def __init__(self, is_dca):
        super(Test_sc6k, self).__init__()
        names = os.listdir(join(DATA_ROOT, 'sc6k'))
        black_list = ['5o31_2_NAP_PROT.pdb']
        names.sort()
        self.protein_paths, self.cavity_paths, self.ligand_paths = [], [], []
        for name in names:
            tmp_protein_paths = glob.glob(join(DATA_ROOT, 'sc6k', name, '{}_*PROT.pdb'.format(name)))
            if name == '5o31':
                tmp_protein_paths = [path for path in tmp_protein_paths if os.path.basename(path) not in black_list]
            tmp_protein_paths.sort()
            self.protein_paths += tmp_protein_paths
            for prot_path in tmp_protein_paths:
                tmp_cavity_paths = glob.glob(prot_path.replace('PROT.pdb', '*_ALL.mol2'))
                tmp_ligand_paths = glob.glob(prot_path.replace('_PROT.pdb', '.mol2'))
                self.cavity_paths.append(tmp_cavity_paths)
                self.ligand_paths.append(tmp_ligand_paths)
        print(len(self.protein_paths), len(self.cavity_paths), len(self.ligand_paths))  # 6389 6389
        self.one_channel = False
        if is_dca:
            self.label_paths = self.ligand_paths
        else:
            self.label_paths = self.cavity_paths

    def __getitem__(self, index):
        protein_path = self.protein_paths[index]
        centroid, mol = self.get_mol(protein_path)
        protein_x = self.density_form_mol(mol, self.one_channel)  # # (18, 36, 36, 36)
        truth_labels, num_cavity = self.get_label_grids(self.label_paths[index], centroid)  # (1, 36, 36, 36, 1)
        truth_labels = np.squeeze(truth_labels)  # # (36, 36, 36)
        return protein_x, truth_labels, centroid, len(self.cavity_paths[index])

    def __len__(self):
        return len(self.protein_paths)
