import os
import random
from torch.utils.data import Dataset as dataset, DataLoader
from utils.common import *
import parameter as para
import matplotlib.pyplot as plt


class Dataset(dataset):
    def __init__(self, dataset_path, model=None):

        self.dataset_path = dataset_path

        if model == 'train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_name_list.txt'))
        elif model == 'val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'val_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')

    def __getitem__(self, index):

        ct_array, seg_array = self.get_train_val_batch_by_index(index=index)

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def get_train_val_batch_by_index(self,index):

        ct = sitk.ReadImage(self.dataset_path + '/ct/' + self.filename_list[index], sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(self.dataset_path + '/seg/' + self.filename_list[index].replace('volume', 'segmentation'), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        # min max 归一化
        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 在slice片面内随机选择48张slice
        start_slice = random.randint(0, ct_array.shape[0] - para.size)
        end_slice = start_slice + para.size - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        return ct_array, seg_array

    def __len__(self):

        return len(self.filename_list)


def main():

    dataset_path = ''
    dataset = Dataset(dataset_path, model='train')
    data_loader = DataLoader(dataset=dataset, batch_size=4, num_workers=1, shuffle=True)
    for i, data_img in enumerate(data_loader):
        raw, label = data_img
        label = to_one_hot_3d(label.long())
        print(raw.shape, label.shape)  # torch.Size([4, 1, 48, 256, 256]) torch.Size([4, 48, 512, 512])
        plt.subplot(121)
        plt.imshow(raw[0, 0, 0])
        plt.subplot(122)
        plt.imshow(label[0, 1, 0])
        plt.show()


if __name__ == '__main__':
    main()