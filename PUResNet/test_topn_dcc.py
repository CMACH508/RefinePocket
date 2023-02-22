# _date_:2021/8/29 16:10
import numpy as np
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
from skimage.draw import ellipsoid
import os, sys
import argparse
from scipy.spatial.distance import cdist
from dataset import TestPDBbind, Test_coach420_holo4k, Test_sc6k
from criterion import dice_loss, ovl, dice
import pybel
from pybel import readfile
from skimage.morphology import binary_dilation
from skimage.morphology import cube
pybel.ob.obErrorLog.SetOutputLevel(0)
# def get_pockets_segmentation(density, threshold=0.5, min_size=50, scale=0.5):
def get_pockets_segmentation(density, threshold=0.5, min_size=50, scale=0.5, max_n = None):
    """Predict pockets using specified threshold on the probability density.
    Filter out pockets smaller than min_size A^3
    """

    if len(density) != 1:
        raise ValueError('segmentation of more than one pocket is not'
                         ' supported')

    voxel_size = (1 / scale) ** 3  # scale?
    # get a general shape, without distinguishing output channels

    bw = closing((density[0] > threshold).any(axis=-1))

    # print('--------------')
    # remove artifacts connected to border
    cleared = clear_border(bw)
    # a = cleared[cleared > 0.5]
    # print(cleared)

    # label regions
    label_image, num_labels = label(cleared, return_num=True)
    # new
    size_list = []
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum() * voxel_size
        if pocket_size < min_size:
            label_image[np.where(pocket_idx)] = 0
            pocket_size = 0
        size_list.append(pocket_size)
    indexs = np.argsort(-np.array(size_list))  # da -- xiao
    indexs = indexs[:max_n] + 1
    label_list = indexs
    new_label_image = np.zeros_like(label_image)
    # print(label_list)
    for ii, lab in enumerate(label_list):
        pocket_idx = (label_image == lab)
        new_label_image[np.where(pocket_idx)] = ii + 1
    return new_label_image
    # old
    # if max_n is None:
    #     for i in range(1, num_labels + 1):
    #         pocket_idx = (label_image == i)
    #         pocket_size = pocket_idx.sum() * voxel_size
    #         if pocket_size < min_size:
    #             label_image[np.where(pocket_idx)] = 0
    #     return label_image
    #
    # else:
    #     size_list = []
    #     for i in range(1, num_labels + 1):
    #         pocket_idx = (label_image == i)
    #         pocket_size = pocket_idx.sum() * voxel_size
    #         # print(i, pocket_size, min_size)
    #         if pocket_size >= min_size:
    #             size_list.append(pocket_size)
    #     indexs = np.argsort(-np.array(size_list))  # da -- xiao
    #     indexs = indexs[:max_n] + 1
    #     label_list = indexs
    #
    #     new_label_image = np.zeros_like(label_image)
    #     # print(label_list)
    #     for ii, lab in enumerate(label_list):
    #         pocket_idx = (label_image == lab)
    #         new_label_image[np.where(pocket_idx)] = ii+1
    #     return new_label_image

def _get_binary_features(mol):
    coords = []
    for a in mol.atoms:
        coords.append(a.coords)
    coords = np.array(coords)
    features = np.ones((len(coords), 1))
    return coords, features

def get_label_grids(cavity_paths):
    # cavitys = glob.glob(join(dir_path, cavity_keyword))
    # print(dir_path, self.suffix)
    # print('len_cavity=', len(cavitys), dir_path)
    # label_grids = np.zeros(shape=(1, 36, 36, 36, 1))
    pocket_number = 0
    pocket_coords_list = []

    # print('cavity_paths=', cavity_paths)

    cavity_suffix = cavity_paths[0].split('.')[-1]
    for n, cavity_path in enumerate(cavity_paths, start=1):
        print('cavity_path=', cavity_path)
        mol = next(readfile(cavity_suffix, cavity_path))
        pocket_coords, pocket_features = _get_binary_features(mol)
        pocket_coords_list.append(pocket_coords)
        pocket_number += 1
    #     x = x * pocket_number
    #     label_grids += x
    #     label_grids = np.where(label_grids > n, n, label_grids)
    #
    # return label_grids.astype(int), pocket_number
    return pocket_coords_list, pocket_number

def test_model(model, device, data_loader, scale, Threshold_dist, test_set=None, is_dca=0, top_n=0):
    model.eval()
    succ = 0
    total = 0
    dvo = 0
    DVO_list = []
    max_n = None
    with torch.no_grad():
        for ite, (protien_x, label, centerid, real_num) in enumerate(data_loader, start=1):
            print('Processing {}/{}'.format(ite, len(data_loader)))
            # print(protien_x.shape, label.shape)
            # centroid = centroid.data.cpu().numpy()
            protien_x, label = protien_x.to(device), label.cpu().numpy()  # (bs, 18, 36, 36, 36) # (bs, 36, 36, 36)
            predy_density = model(protien_x)  # (bs, 1, 36, 36, 36)
            # print(label_01.shape, label.shape, predy_density.shape)
            # tot_loss = dice_loss(y_pred=predy_density, y_true=label_01)
            # print('loss=', tot_loss.item())

            predy_density = predy_density.data.cpu().numpy()

            for i, density in enumerate(predy_density):  # (1, 36, 36, 36)

                total += real_num[i]
                density = np.expand_dims(density, 4)  # (1, 36, 36, 36, 1)
                truth_labels = label[i]
                num_cavity = int(truth_labels.max())
                # print('num_cavity={}'.format(num_cavity))
                if num_cavity == 0:
                    # print('--------========xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    continue

                max_n = real_num[i] + top_n
                # if top_n in [0, 2]:
                #     max_n = real_num[i] + top_n
                # else:
                #     max_n = None
                # if is_dca:
                #     max_n = real_num[i]
                #     # max_n = real_num[i] + 2
                # else:
                #     max_n = None
                # # print('max_n=', max_n)

                predict_labels = get_pockets_segmentation(density, scale=scale, max_n=max_n)

                for target_num in range(1, num_cavity + 1):
                    truth_indices = np.argwhere(truth_labels == target_num).astype('float32')
                    label_center = truth_indices.mean(axis=0)
                    min_dist = 1e6
                    match_label = 0
                    # print('predict_labels num ====', predict_labels.max())
                    # for m in range(1, predict_labels.max()+1):
                    #     print((predict_labels == m).any())
                    # print('  pred num={} pocket'.format(predict_labels.max()))
                    for pocket_label in range(1, predict_labels.max() + 1):
                        indices = np.argwhere(predict_labels == pocket_label).astype('float32')
                        if len(indices) == 0:
                            continue
                        #     print('label=', pocket_label)
                        center = indices.mean(axis=0)

                        if is_dca:
                            dist = 1e6
                            for c in truth_indices:
                                d = np.linalg.norm(center - np.array(c))
                                if d < dist:
                                    dist = d
                        else:
                            dist = np.linalg.norm(label_center - center)

                        if dist < min_dist:
                            min_dist = dist
                            match_label = pocket_label

                    if min_dist <= Threshold_dist:
                        # print(min_dist, Threshold_dist)
                        succ += 1
                        indices = np.argwhere(predict_labels == match_label).astype('float32')

                        if test_set == 'pdbbind':
                            protien_array = protien_x[i].data.cpu().numpy()  # (18,36,36,36)
                            protien_array = protien_array.transpose((1, 2, 3, 0))  # (36,36,36,18)
                            protein_coord = []
                            for k1 in range(36):
                                for k2 in range(36):
                                    for k3 in range(36):
                                        # print(protien_array[k1, k2, k3].shape)  #(18)
                                        # if not np.all(protien_array[k1, k2, k3]):
                                        if np.any(protien_array[k1, k2, k3]):
                                            protein_coord.append(np.asarray([k1, k2, k3]))
                            protein_coord = np.asarray(protein_coord)
                            ligand_dist = cdist(indices, protein_coord)
                            # print('===========')
                            # print(indices.shape)  # (60, 3)
                            # print(protein_coord.shape)  # (46656, 3)
                            # print(ligand_dist.shape)  # (60, 46656)
                            distance = 3
                            # distance = 1.5
                            binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
                            # print(binding_indices[0].shape)
                            # print(protein_coord[binding_indices].shape)
                            indices = protein_coord[binding_indices]
                            # print(binding_indices[0])
                            # cc = 0
                            # for coor in truth_indices:
                            #     if coor in protein_coord:
                            #         # print(coor)
                            #         cc += 1
                            # print('cc=', cc, len(truth_indices))
                            ligand_dist = cdist(truth_indices, protein_coord)
                            distance = 1
                            # distance = 0.5
                            binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
                            truth_indices = protein_coord[binding_indices]


                        indices_set = set([tuple(x) for x in indices])
                        truth_indices_set = set([tuple(x) for x in truth_indices])

                        # print(len(indices_set), len(truth_indices_set))

                        dvo = len(indices_set & truth_indices_set) / len(indices_set | truth_indices_set)
                        DVO_list.append(dvo)


            print('{} now: succ={} total={} succ/total={} dvo={} ({})'.format(test_set, succ, total, succ / total,
                                                                           (np.sum(DVO_list)/total), np.mean(DVO_list)))

        # print('len_dvo=', len(DVO_list), np.sum(DVO_list), np.mean(DVO_list))
        res = '| succ={} | total={} | succ/total={} | dvo={} ({})'.format(succ, total, succ / total,
                                                                          (np.sum(DVO_list)/total), np.mean(DVO_list))
        return res

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--is_dca', type=int, default=0)
parser.add_argument('-n', '--top_n', type=int, default=0)
parser.add_argument('--DATA_ROOT', type=str, default='DATA_ROOT')
parser.add_argument('--test_set', type=str, default='coach420') # coach420,sc6k,holo4k,pdbbind,apoholo
parser.add_argument('--model_path', type=str)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda')
    dataset = None

    mask = False
    one_channel = False
    test_set = args.test_set
    is_dca = args.is_dca
    model_path = args.model_path
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if test_set == 'scpdb':
        dataset = TestscPDB(one_channel=False, mask=mask)
    elif test_set == 'pdbbind':
        dataset = TestPDBbind(one_channel=False, mask=mask, is_dca=is_dca)
    elif test_set == 'apo_holo':
        dataset = TestApoHolo(one_channel=False, mask=mask, is_dca=is_dca)
    elif test_set == 'coach420' or test_set == 'holo4k':
        dataset = Test_coach420_holo4k(set=test_set, is_dca=is_dca)
    elif test_set == 'sc6k':
        dataset = Test_sc6k(is_dca=is_dca)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=6)

    print('Restoring model from path: ' + model_path)

    from ResNet36 import PUResNet

    # TODO which model
    model = PUResNet().to(device)
    model = DataParallel(model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    res = test_model(model, device, data_loader, scale=0.5, Threshold_dist=2, test_set=test_set, is_dca=is_dca, top_n=args.top_n)
    print('\nT={} | {} {}'.format(2, test_set, res))
    print('model_path:', model_path)
    print('test_set:', test_set)
    print('is_dca:', is_dca)
    print('top-n:', args.top_n)

    # file = open('/cmach-data/lipeiying/_Drug_/binding_site/baseline_{}_dcc_dvo.txt'.format(test_set), 'w')
    # file.write(model_path+'\n')
    # for T in range(1, 21):
    #     T = T * 0.5
    #     res = test_model(model, device, data_loader, scale=0.5, Threshold_dist=T)
    #     print('T={} | {} {}'.format(T, test_set, res))
    #     print(model_path)
    #     file.write('T={} | {} {}\n'.format(T, test_set, res))
    # print('------------ Finish -----------')
