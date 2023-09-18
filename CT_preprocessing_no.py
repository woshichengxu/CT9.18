#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
#import random
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage



class CTDataset(Dataset):

    def __init__(self, root_dir, img_list, sets, phase):
        self.img_list=[]  #此处进行了修改记得检查
        with open(img_list, 'r',encoding="utf-8") as f:
            # print(f)
            self.img_list = [line.strip() for line in f]
            # print(self.img_list)
        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = phase
        self.batch_size = sets.batch_size
        self.classes = sets.n_cls_classes

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        if self.phase == "train":
            # read image and labels
            ith_info = self.img_list[idx].split(" ")

            img_name = ith_info[0]
            label = int(ith_info[1])-1 

            assert os.path.isfile(img_name)

            img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW

            assert img is not None
            assert label is not None
            #log.info(img_name)
            #log.info(img.shape)
            # data processing
            img_array = self.__training_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)

            # one_hot = torch.zeros(self.classes).scatter_(dim=0, index = torch.tensor(label)-1, value=1.0)
            # one_hot = np.eye(self.classes)[label-1]
            # one_hot = torch.zeros(self.batch_size, self.classes)
            # assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            return img_array, label

        elif self.phase == "test":
            # read image
            ith_info = self.img_list[idx].split(" ")

            img_name = ith_info[0]
            label = int(ith_info[1])-1

            assert os.path.isfile(img_name)

            img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW

            assert img is not None
            assert label is not None

            # data processing
            img_array = self.__testing_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)

            #one_hot = np.eye(self.classes)[label-1]
            # one_hot = torch.zeros(self.batch_size, self.classes)
            return img_array, label

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)
        if np.array(non_zeros_idx).size == 0:
            return volume

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]


    def __training_data_process__(self, data):
        # crop data according net input size
        data = data.get_fdata()

        return data

    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_fdata()

        return data


# In[ ]:


