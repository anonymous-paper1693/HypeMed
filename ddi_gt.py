
import dill
import os

from util import ddi_rate_score

from tqdm import tqdm

def dill_load(pth, mode='rb'):
    with open(pth, mode) as fr:
        file = dill.load(fr)
    return file

def load_data(mimic=3):
    if mimic == 3:
        # load old_data
        data_path = './data/records_final.pkl'
        voc_path = './data/voc_final.pkl'

        # ehr_adj_path = '../data/ehr_adj_final.pkl'
        ddi_adj_path = './data/ddi_A_final.pkl'
        # ddi_mask_path = '../data/ddi_mask_H.pkl'
        # molecule_path = '../data/idx2drug.pkl'
    elif mimic == 4:
        # load old_data
        data_path = './data/records_final_4.pkl'
        voc_path = './data/voc_final_4.pkl'

        # ehr_adj_path = '../data/ehr_adj_final.pkl'
        ddi_adj_path = './data/ddi_A_final_4.pkl'
        # ddi_mask_path = '../data/ddi_mask_H.pkl'
        # molecule_path = '../data/idx2drug.pkl'
    else:
        raise ValueError('Wrong Dataset Arg.!!!!!')

    if mimic == 3:
        cache_dir = './data/cache'
    else:
        cache_dir = './data/cache_4'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    data = dill_load(data_path, "rb")
    assert len(data) > 100
    return data, ddi_adj_path

def compute_ddi_gt(data_test, ddi_adj_path):
    medications = []
    for patient in tqdm(data_test):
        cur_med = []
        for adm in patient:
            med = adm[2]
            cur_med.append(med)
        medications.append(cur_med)

    ddi = 0.
    for i in range(len(medications)):
        ddi += ddi_rate_score([medications[i]], ddi_adj_path)
    return ddi / len(medications)

if __name__ == '__main__':
    data_test, ddi_adj_path = load_data(4)
    ddi = compute_ddi_gt(data_test, ddi_adj_path)
    print(ddi)
    # mimic 3 0.08683760238904287
    # mimic 4 0.07240760531487284