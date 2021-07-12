import numpy as np
import os
import SimpleITK as sitk
import random
from tqdm import tqdm
from scipy import ndimage
import parameter as para


def generate_clock(image, array, start, end):
    """
    生成厚度为16的切片块
    :param image: 原始的volume.nii或segmentation.nii所对应的image
    :param array: 原始的volume.nii或segmentation.nii所对应的array
    :param start: 切片块的起始下标
    :param end:   切片块的终止下标
    :return: 返回切片块对应的image文件
    """
    array_block = array[start:end + 1, :, :] if end else array[start:, :, :]
    image_block = sitk.GetImageFromArray(array_block)
    image_block.SetDirection(image.GetDirection())
    image_block.SetOrigin(image.GetOrigin())
    image_block.SetSpacing(
        (image.GetSpacing()[0] * int(1 / para.down_scale),
         image.GetSpacing()[1] * int(1 / para.down_scale),
         para.slice_thickness)
    )

    return image_block


class LITS_fix:
    def __init__(self, raw_dataset_path, fixed_dataset_path):
        # self.raw_root_path = para.raw_data_path
        # self.fixed_path = para.fix_data_path

        self.raw_root_path = para.raw_liver_data_path
        self.fixed_path = para.fix_liver_data_path

        if not os.path.exists(self.fixed_path):  # 创建保存目录
            os.makedirs(self.fixed_path + 'ct')
            os.makedirs(self.fixed_path + 'seg')

        self.fix_data()  # 对原始图像进行修剪并保存
        self.write_train_val_test_name_list()  # 创建索引txt文件

    def fix_data(self):

        print('the raw dataset total numbers of samples is :', len(os.listdir(self.raw_root_path + 'ct')))
        idx = 0  # 切片块文件编号
        for ct_file in tqdm(os.listdir(self.raw_root_path + 'ct/')):

            print(ct_file)

            # 将CT和金标准入读内存
            ct = sitk.ReadImage(os.path.join(self.raw_root_path + 'ct/', ct_file), sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)
            # 将CT和金标准入读内存
            seg = sitk.ReadImage(os.path.join(self.raw_root_path + 'seg/', ct_file.replace('volume', 'segmentation')),
                                 sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            # 将金标准中肝脏和肿瘤的标签融合为一个
            seg_array[seg_array > 0] = 1

            # 将灰度值在阈值之外的截断掉
            ct_array[ct_array > para.upper] = para.upper
            ct_array[ct_array < para.lower] = para.lower

            # 对CT数据在横断面上进行降采样(下采样),并进行重采样,将所有数据的z轴的spacing调整到1mm
            ct_array = ndimage.zoom(ct_array,
                                    (1, para.down_scale, para.down_scale),
                                    order=1)
            seg_array = ndimage.zoom(seg_array, (1, 1,  1), order=0)

            # 找到肝脏区域开始和结束的slice，并各向外扩张
            z = np.any(seg_array, axis=(1, 2))
            start_slice, end_slice = np.where(z)[0][[0, -1]]

            # 两个方向上各扩张个slice
            start_slice = max(0, start_slice - para.expand_slice)
            end_slice = min(seg_array.shape[0] - 1, end_slice + para.expand_slice)

            print(str(start_slice) + '--' + str(end_slice))
            # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
            if end_slice - start_slice + 1 < para.size:
                print(ct_file, 'too little slice，give up the sample')
                continue

            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]

            # 开始生成厚度为48的切片块，并写入文件中，保存为nii格式
            l, r = 0, para.block_size - 1
            while r < ct_array.shape[0]:
                # volume切片块和segmentation切片块生成
                ct_block = generate_clock(ct, ct_array, l, r)
                seg_block = generate_clock(seg, seg_array, l, r)

                ct_block_name = 'volume-' + str(idx) + '.nii'
                seg_block_name = 'segmentation-' + str(idx) + '.nii'
                sitk.WriteImage(ct_block, os.path.join(para.fix_liver_data_path + '/ct', ct_block_name))
                sitk.WriteImage(seg_block, os.path.join(para.fix_liver_data_path + '/seg', seg_block_name))

                idx += 1
                l += para.stride
                r = l + para.block_size - 1

            # 如果每隔opt.stride不能完整的将所有切片分块时，从后往前取到最后一个block
            if r != ct_array.shape[0] + para.stride_1:
                # volume切片块生成
                ct_block = generate_clock(ct, ct_array, -para.block_size, None)
                seg_block = generate_clock(seg, seg_array, -para.block_size, None)

                ct_block_name = 'volume-' + str(idx) + '.nii'
                seg_block_name = 'segmentation-' + str(idx) + '.nii'
                sitk.WriteImage(ct_block, os.path.join(para.fix_liver_data_path + '/ct', ct_block_name))
                sitk.WriteImage(seg_block, os.path.join(para.fix_liver_data_path + '/seg', seg_block_name))

                idx += 1

    def write_train_val_test_name_list(self):
        data_name_list = os.listdir(self.fixed_path + "/" + "ct")
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        train_rate = 0.6
        val_rate = 0.4
        # test_rate = 0.1

        assert val_rate + train_rate == 1.0
        train_name_list = data_name_list[0:int(data_num * train_rate)]
        val_name_list = data_name_list[int(data_num * train_rate):int(data_num * (train_rate + val_rate))]
        # test_name_list = data_name_list[int(data_num * (train_rate + val_rate)):len(data_name_list)]

        self.write_name_list(train_name_list, "train_name_list.txt")
        self.write_name_list(val_name_list, "val_name_list.txt")
        # self.write_name_list(test_name_list, "test_name_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(self.fixed_path + file_name, 'w')
        for i in range(len(name_list)):
            f.write(str(name_list[i]) + "\n")
        f.close()


def main():
    raw_dataset_path = para.raw_liver_data_path
    fixed_dataset_path = para.fix_liver_data_path

    LITS_fix(raw_dataset_path, fixed_dataset_path)


if __name__ == '__main__':
    main()
