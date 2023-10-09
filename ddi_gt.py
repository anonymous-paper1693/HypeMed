
import dill
import os

from util import ddi_rate_score

def dill_load(pth, mode='rb'):
    with open(pth, mode) as fr:
        file = dill.load(fr)
    return file

def load_data(mimic=3):
    if mimic == 3:
        # load old_data
        data_path = '../data/records_final.pkl'
        voc_path = '../data/voc_final.pkl'

        # ehr_adj_path = '../data/ehr_adj_final.pkl'
        ddi_adj_path = '../data/ddi_A_final.pkl'
        # ddi_mask_path = '../data/ddi_mask_H.pkl'
        # molecule_path = '../data/idx2drug.pkl'
    elif mimic == 4:
        # load old_data
        data_path = '../data/records_final_4.pkl'
        voc_path = '../data/voc_final_4.pkl'

        # ehr_adj_path = '../data/ehr_adj_final.pkl'
        ddi_adj_path = '../data/ddi_A_final_4.pkl'
        # ddi_mask_path = '../data/ddi_mask_H.pkl'
        # molecule_path = '../data/idx2drug.pkl'
    else:
        raise ValueError('Wrong Dataset Arg.!!!!!')

    if mimic == 3:
        cache_dir = '../data/cache'
    else:
        cache_dir = '../data/cache_4'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = cache_dir
    # desc_dict_path = '../data/desc_dict.pkl'

    ddi_adj = dill_load(ddi_adj_path, "rb")
    # ddi_mask_H = dill_load(ddi_mask_path, "rb")
    data = dill_load(data_path, "rb")
    assert len(data) > 100
    # molecule = dill_load(molecule_path, "rb")
    # desc_dict = dill_load(desc_dict_path, 'rb')
    voc = dill_load(voc_path, "rb")

    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    # 这里数据划分好像有点问题，不是按照每个病人划分的，也没有shuffle
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    assert eval_len > 100
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    return data_test, ddi_adj_path

def compute_ddi_gt(data_test, ddi_adj_path):
    medications = []
    for patient in data_test:
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
    # mimic 3 0.08100028033015957
    # mimic 4 0.07294106549352017