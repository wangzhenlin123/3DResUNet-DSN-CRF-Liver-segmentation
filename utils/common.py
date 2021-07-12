import SimpleITK as sitk
import numpy as np
import torch


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list


# 将CT和金标准读入到内存中
def sitk_read_raw(img_path):
    ct = sitk.ReadImage(img_path, sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 20

    seg = sitk.ReadImage(img_path, sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    return ct_array, seg_array


# target one-hot编码
def to_one_hot_3d(tensor, n_classes=2):  # shape = [batch, s, h, w]
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot


# 训练时的Dice指标
def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    dice = (2. * inter + 1) / (union + 1)
    return dice


def accuracy(result, reference):
    result = torch.argmax(result,dim=1).cpu().numpy()
    result = np.atleast_1d(result.astype(np.bool))
    reference = reference.cpu().numpy()
    reference = np.atleast_1d(reference.astype(np.bool))
    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)
    tn = np.count_nonzero(~result & ~reference)
    fn = np.count_nonzero(~result & reference)
    try:
        acc = (tp+tn) / float(tp + fp + tn + fn)
    except ZeroDivisionError:
        acc = 0.0
    return acc
