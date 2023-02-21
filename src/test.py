import datetime
from os.path import join
import torch
import numpy as np
import argparse
import os
import molgrid
from skimage.morphology import closing
from skimage.segmentation import clear_border
from skimage.measure import label
from scipy.spatial.distance import cdist
from skimage.morphology import binary_dilation
from skimage.morphology import cube
import prody
# from pybel import readfile
from openbabel import pybel
import glob

prody.confProDy(verbosity='none')
pybel.ob.obErrorLog.SetOutputLevel(0)

def preprocess_output(input, threshold):
    input[input >= threshold] = 1
    input[input != 1] = 0
    input = input.numpy()
    bw = closing(input).any(axis=0)
    # remove artifacts connected to border
    cleared = clear_border(bw)

    # label regions
    label_image, num_labels = label(cleared, return_num=True)
    largest = 0
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum()
        if pocket_size > largest:
            largest = pocket_size
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum()
        if pocket_size < largest:
            label_image[np.where(pocket_idx)] = 0
    label_image[label_image > 0] = 1
    return torch.tensor(label_image, dtype=torch.float32)


def get_model_gmaker_eproviders(args):
    # test example provider
    eptest = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, iteration_scheme=molgrid.IterationScheme.LargeEpoch,
                                     default_batch_size=1)
    eptest.populate(args.test_types)
    # gridmaker with defaults
    gmaker_img = molgrid.GridMaker(dimension=32)

    return gmaker_img, eptest


def Output_Coordinates(tensor, center, dimension=16.25, resolution=0.5):
    # get coordinates of mask from predicted mask
    tensor = tensor.numpy()
    indices = np.argwhere(tensor > 0).astype('float32')
    indices *= resolution
    center = np.array([float(center[0]), float(center[1]), float(center[2])])
    indices += center
    indices -= dimension
    return indices


def output_pocket_pdb(pocket_name, prot_prody, pred_AA):
    # output pocket pdb
    # skip if no amino acids predicted
    if len(pred_AA) == 0:
        return
    sel_str = 'resindex '
    for i in pred_AA:
        sel_str += str(i) + ' or resindex '
    sel_str = ' '.join(sel_str.split()[:-2])
    pocket = prot_prody.select(sel_str)
    prody.writePDB(pocket_name, pocket)


def _get_binary_features(mol):
    coords = []
    for a in mol.atoms:
        coords.append(a.coords)
    coords = np.array(coords)
    features = np.ones((len(coords), 1))
    return coords, features


def get_label_grids(cavity_paths):
    pocket_number = 0
    pocket_coords_list = []
    cavity_suffix = cavity_paths[0].split('.')[-1]
    for n, cavity_path in enumerate(cavity_paths, start=1):
        mol = next(pybel.readfile(cavity_suffix, cavity_path))
        pocket_coords, pocket_features = _get_binary_features(mol)
        pocket_coords_list.append(pocket_coords)
        pocket_number += 1
    return pocket_coords_list, pocket_number


def get_model_gmaker_eprovider(test_types, batch_size, dims=None):
    eptest_large = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, labelpos=0, balanced=False,
                                           iteration_scheme=molgrid.IterationScheme.LargeEpoch, default_batch_size=batch_size)
    eptest_large.populate(test_types)
    if dims is None:
        gmaker = molgrid.GridMaker()
    else:
        gmaker = molgrid.GridMaker(dimension=dims)
    return gmaker, eptest_large


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def union(lst1, lst2):
    return list(set().union(lst1, lst2))


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def binding_site_AA(ligand, prot_prody, distance):
    # amino acids from ligand distance threshold
    c = ligand.GetConformer()
    ligand_coords = c.GetPositions()

    prot_coords = prot_prody.getCoords()
    ligand_dist = cdist(ligand_coords, prot_coords)
    binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
    # Get protein residue indices involved in binding site
    prot_resin = prot_prody.getResindices()
    prot_binding_indices = prot_resin[binding_indices]
    prot_binding_indices = sorted(list(set(prot_binding_indices)))
    return prot_binding_indices


def predicted_AA(indices, prot_prody, distance):
    # amino acids from mask distance thresholds
    prot_coords = prot_prody.getCoords()
    ligand_dist = cdist(indices, prot_coords)
    binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
    # get predicted protein residue indices involved in binding site
    prot_resin = prot_prody.getResindices()
    prot_binding_indices = prot_resin[binding_indices]
    prot_binding_indices = sorted(list(set(prot_binding_indices)))
    return prot_binding_indices


def coors2grid(coords, box_size=80):
    grid = np.zeros(shape=(box_size, box_size, box_size))
    center = coords.mean(axis=0)
    coords -= center
    coords += (box_size / 2)
    coords = coords.round().astype(int)

    in_box = ((coords >= 0) & (coords < box_size)).all(axis=1)

    for (x, y, z) in coords[in_box]:
        grid[x, y, z] = 1

    return grid, center

def get_pdbbind(DATA_ROOT):
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

    protein_paths = glob.glob(path)
    protein_paths.sort()
    print(len(protein_paths))
    protein_paths = [p for p in protein_paths if os.path.basename(p) not in total_list]
    cavity_paths = [[path.replace('protein', 'pocket')] for path in protein_paths]
    ligand_paths = [[path.replace('protein.pdb', 'ligand.mol2')] for path in protein_paths]
    print('all_data=', len(protein_paths), len(cavity_paths), len(ligand_paths))
    return protein_paths, cavity_paths, ligand_paths


def get_sc6k(DATA_ROOT):
    names = os.listdir(join(DATA_ROOT, 'sc6k'))
    black_list = ['5o31_2_NAP_PROT.pdb']
    names.sort()
    protein_paths, cavity_paths, ligand_paths = [], [], []
    for name in names:
        tmp_protein_paths = glob.glob(join(DATA_ROOT, 'sc6k', name, '{}_*PROT.pdb'.format(name)))
        if name == '5o31':
            tmp_protein_paths = [path for path in tmp_protein_paths if os.path.basename(path) not in black_list]
        tmp_protein_paths.sort()
        protein_paths += tmp_protein_paths
        for prot_path in tmp_protein_paths:
            tmp_cavity_paths = glob.glob(prot_path.replace('PROT.pdb', '*_ALL.mol2'))
            tmp_ligand_paths = glob.glob(prot_path.replace('_PROT.pdb', '.mol2'))
            cavity_paths.append(tmp_cavity_paths)
            ligand_paths.append(tmp_ligand_paths)
    print(len(protein_paths), len(cavity_paths), len(ligand_paths))  # 6389 6389
    return protein_paths, cavity_paths, ligand_paths



def get_apoholo(DATA_ROOT, is_dca=0):
    suffix = 'pdb'
    # suffix = 'mol2'
    apo_root = join(DATA_ROOT, 'APO-volsite-cavity')
    holo_root = join(DATA_ROOT, 'HOLO-volsite-cavity')
    ligand_dir = join(DATA_ROOT, 'apo_and_holo', 'holo-split-ligand')
    protein_paths = []
    cavity_paths = []
    ligand_paths = []
    cnt_ligand = 0
    cavity_match_word = 'CAVITY_*ALL.mol2'

    apo_protein_dirs = glob.glob(join(apo_root, '*'))
    apo_protein_dirs.sort()
    for d in apo_protein_dirs:
        tmp_cavity_paths = glob.glob(join(d, cavity_match_word))
        cavity_len = len(tmp_cavity_paths)
        if cavity_len > 0:
            protein_path = glob.glob(join(d, '*_unbound.' + suffix))[0]  # a.102.001.004_1v7va_unbound.mol2

            if is_dca:
                ligand_path = glob.glob(join(ligand_dir, os.path.basename(protein_path).rsplit('_', 1)[0]+'*'))
                if len(ligand_path) > 0:
                    cnt_ligand += len(ligand_path)
                    ligand_paths.append(ligand_path)
                    protein_paths.append(protein_path)
                    cavity_paths.append(tmp_cavity_paths)
                else:
                    continue
            else:
                protein_paths.append(protein_path)
                cavity_paths.append(tmp_cavity_paths)

    holo_protein_dirs = glob.glob(join(holo_root, '*'))
    holo_protein_dirs.sort()
    for d in holo_protein_dirs:
        tmp_cavity_paths = glob.glob(join(d, cavity_match_word))
        cavity_len = len(tmp_cavity_paths)
        if cavity_len > 0:
            protein_path = glob.glob(join(d, '*_protein.' + suffix))[0]

            if is_dca:
                ligand_path = glob.glob(join(ligand_dir, os.path.basename(protein_path).rsplit('_', 1)[0] + '*'))
                if len(ligand_path) > 0:
                    cnt_ligand += len(ligand_path)
                    ligand_paths.append(ligand_path)
                    protein_paths.append(protein_path)
                    cavity_paths.append(tmp_cavity_paths)
                else:
                    continue
            else:
                protein_paths.append(protein_path)
                cavity_paths.append(tmp_cavity_paths)

    print(len(protein_paths), len(cavity_paths), len(ligand_paths), 'is_dca={}'.format(is_dca), 'cnt_ligand={}'.format(cnt_ligand))
    return protein_paths, cavity_paths, ligand_paths


def get_coach420_or_holo4k(set, DATA_ROOT):
    protein_root = join(DATA_ROOT, '{}/protein/'.format(set))
    cavity_root = join(DATA_ROOT, '{}/cavity/'.format(set))
    ligand_root = None
    if set == 'coach420':
        ligand_root = join(DATA_ROOT, '{}/ligand_T2_cavity/'.format(set))
    elif set == 'holo4k':
        ligand_root = join(DATA_ROOT, '{}/ligand/'.format(set))
    exist_id = os.listdir(cavity_root)
    exist_id.sort()
    protein_paths = [join(protein_root, '{}.pdb'.format(id_)) for id_ in exist_id]
    cavity_paths = []
    ligand_paths = []
    for id_ in exist_id:
        tmp_cavity_paths = glob.glob(join(cavity_root, id_, '*', 'CAVITY*'))
        if set == 'coach420':
            tmp_ligand_paths = glob.glob(join(ligand_root, id_, 'ligand*'))
        elif set == 'holo4k':
            tmp_ligand_paths = glob.glob(join(cavity_root, id_, '*', 'ligand*'))
        cavity_paths.append(tmp_cavity_paths)
        ligand_paths.append(tmp_ligand_paths)
    print(len(protein_paths), len(cavity_paths), len(ligand_paths))
    return protein_paths, cavity_paths, ligand_paths

dvo_dict = {}
def test(is_dca, protein_path, label_paths, model, test_loader, gmaker_img, device, dx_name, args):
    label_coords_list, num_cavity = get_label_grids(label_paths)  # (1, 36, 36, 36, 1)

    count = 0
    model.eval()

    dims = gmaker_img.grid_dimensions(test_loader.num_types())
    tensor_shape = (1,) + dims
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device=device, requires_grad=True)
    float_labels = torch.zeros((1, 4), dtype=torch.float32, device=device)
    prot_prody = prody.parsePDB(protein_path)
    pred_aa_list = []
    pred_pocket_coords_list = []
    pred_pocket_ii_list = []

    if is_dca:
        max_n = num_cavity + args.rank
    if is_dca and test_set == 'apoholo':
        max_n = 100

    for ii, batch in enumerate(test_loader):
        batch.extract_labels(float_labels)
        centers = float_labels[:, 1:]
        for b in range(1):
            center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
            gmaker_img.forward(center, batch[b].coord_sets[0], input_tensor[b])
        with torch.no_grad():
            masks_pred = model(input_tensor[:, :14])
        masks_pred = masks_pred.detach().cpu()
        masks_pred = preprocess_output(masks_pred[0], args.threshold)

        pred_coords = Output_Coordinates(masks_pred, center)  
        pred_aa = predicted_AA(pred_coords, prot_prody, args.mask_dist)  
        if not len(pred_aa) == 0:
            pred_pocket_coords_list.append(pred_coords)
            pred_pocket_ii_list.append(ii)
            count += 1
        if is_dca and count >= max_n:  # DCA
            break


    DVO_list = []
    succ = 0

    for k, label_coords in enumerate(label_coords_list):
        min_dist = 1e6
        match_pred_pocket_coords = None
        ii_ = -1
        for j, pred_pocket_coords in enumerate(pred_pocket_coords_list):
            pred_center_coords = pred_pocket_coords.mean(axis=0)  # (3,)
            if is_dca:
                dist = 1e6
                for c in label_coords:
                    d = np.linalg.norm(pred_center_coords - np.array(c))
                    if d < dist:
                        dist = d
            else:
                dist = np.linalg.norm(pred_center_coords - label_coords.mean(axis=0))

            if dist < min_dist:
                min_dist = dist
                match_pred_pocket_coords = pred_pocket_coords
                ii_ = pred_pocket_ii_list[j]

        if is_dca:
            if min_dist <= 4:
                succ += 1
        else:
            if min_dist <= 4:
                succ += 1
            dvo = 0
            if args.is_dvo:
                if min_dist <= 4:
                    box_size = 80
                    label_grid, label_center = coors2grid(label_coords, box_size=box_size)
                    grid_np = binary_dilation(label_grid, cube(3))
                    grid_indices = np.argwhere(grid_np == 1)
                    label_coords = grid_indices - (box_size / 2)
                    label_coords += label_center

                    pred_coords_set = set([tuple(x) for x in (match_pred_pocket_coords / 2).astype(int)])
                    truth_coords_set = set([tuple(x) for x in (label_coords / 2).astype(int)])
                    dvo = len(pred_coords_set & truth_coords_set) / len(pred_coords_set | truth_coords_set)
                    dvo_dict[os.path.basename(protein_path).split('.')[0] + str(ii_)] = round(dvo, 4)
            DVO_list.append(dvo)

    return succ, num_cavity, DVO_list


def get_acc(seg_model, DATA_ROOT, args, test_set='coach420', is_dca=0):
    protein_paths, cavity_paths, ligand_paths, label_paths = None, None, None, None
    if test_set in ['coach420', 'holo4k']:
        protein_paths, cavity_paths, ligand_paths = get_coach420_or_holo4k(test_set, DATA_ROOT=DATA_ROOT)
    elif test_set == 'pdbbind':
        protein_paths, cavity_paths, ligand_paths = get_pdbbind(DATA_ROOT=DATA_ROOT)
    elif test_set == 'sc6k':
        protein_paths, cavity_paths, ligand_paths = get_sc6k(DATA_ROOT=DATA_ROOT)
    elif test_set == 'apoholo':
        protein_paths, cavity_paths, ligand_paths = get_apoholo(DATA_ROOT=DATA_ROOT, is_dca=is_dca)

    if is_dca:
        label_paths = ligand_paths
    else:
        label_paths = cavity_paths
    save_dir = os.path.join(DATA_ROOT, 'test_types', test_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0
    succ = 0
    dvo_list = []
    length = len(protein_paths)

    for i, protein_path in enumerate(protein_paths):
        if args.is_debug:
            print('{} [{}/{}] {}'.format(get_time(), i, length, protein_path))
        protein_nowat_file = protein_path.replace('.pdb', '_nowat.pdb')
        protein_name = os.path.basename(protein_path)
        id = protein_name.rsplit('.', 1)[0]
        tmp_dir = join(save_dir, id)
        seg_types = '{}/{}_nowat_out/pockets/bary_centers_ranked.types'.format(tmp_dir, id)
        if not os.path.exists(seg_types):
            print('seg_types not exist path=', seg_types)
            break

        lines = open(seg_types).readlines()
        if len(lines) == 0:
            total += len(label_paths[i])
            continue

        seg_gmaker, seg_eptest = get_model_gmaker_eprovider(seg_types, 1, dims=32)
        dx_name = protein_nowat_file.replace('.pdb', '')

        tmp_succ, num_cavity, DVO_list = test(is_dca, protein_path, label_paths[i], seg_model, seg_eptest, seg_gmaker,
                                              device, dx_name, args)
        total += num_cavity
        succ += tmp_succ
        dvo_list += DVO_list
        total = max(total, 0.0000001)
        if args.is_debug:
            print('tmp {}: succ={}, total={}, dcc={}/{}={:.4f}, dvo={:.4f}'.format(
                test_set, succ, total, succ, total, succ / total, np.mean(DVO_list)))

    total = max(total, 0.0000001)
    print('{}: succ={}, total={}, dcc={}/{}={:.4f}, dvo={:.4f}'.format(test_set, succ, total, succ, total, succ / total, np.mean(dvo_list)))
    return succ / total


if __name__ == '__main__':
    from torch import nn
    import argparse, sys
    sys.path.append(os.path.abspath('../'))

    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('--rank', type=int, help="training types file", default=1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mask_dist', type=float, default=3.5)
    # todo modify
    parser.add_argument('--is_dvo', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--gpu', type=str, default='2,3')
    parser.add_argument('--is_mask', type=int, default=0)
    parser.add_argument('--is_seblock', type=int, default=0)
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--is_debug', type=int, default=1)

    parser.add_argument('--is_dca', type=int, default=0)
    parser.add_argument('--test_set', type=str, default='coach420')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from net import Unet
    model = Unet(n_classes=1, upsample=False)
    model.to(device)
    model = nn.DataParallel(model)

    test_set = args.test_set
    is_dca = args.is_dca
    DATA_ROOT = os.path.abspath('/cmach-data/lipeiying/program/_Drug_/dataset/')
    ckpt_path = 'archive3/model_saves/seg0-0.93548-123.pth.tar'

    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    print('load successfully,', ckpt_path)
    a = datetime.datetime.now()

    get_acc(model, DATA_ROOT, args, test_set=test_set, is_dca=is_dca)
    