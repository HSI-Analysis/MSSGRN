import torch
import numpy as np
import scipy.io as scio

import cv2
import sys

class load():
    # load dataset(indian_pines & pavia_univ.)
    def load_data(self, flag='indian', number = '1', trained_on = 'AA'):
        #Changed____________________________________________________________________________________________________________________________________________
        if flag == 'afeyan':
            path_to_folder = "../../HSI-Data/data_size=696x520/"
            type = "porcine" 
            path = path_to_folder + type + number + "_696x520x31/MAT/"
            afeyan_dict = scio.loadmat(path + "data/" + type + number + "_696x520x31.mat")
            afeyan_gt_dict_train = scio.loadmat(path + "gt/" + type + number + "_" + trained_on + "_696x520_gt_m.mat")
            afeyan_gt_dict_test = scio.loadmat(path + "gt/" + type + number + "_not_" + trained_on + "_696x520_gt_m.mat")

            afeya_shape = afeyan_dict[list(afeyan_dict.keys())[3]].shape
            afeyan_new = np.zeros((afeya_shape[1], afeya_shape[2], afeya_shape[0])).astype(np.int64)

            for i in range(31):
                afeyan_new[:,:,i]  = afeyan_dict[list(afeyan_dict.keys())[3]].astype(np.int64)[i, :, :]
            
            afeyan_dict[list(afeyan_dict.keys())[3]] = afeyan_new
            print(afeyan_dict[list(afeyan_dict.keys())[3]].shape)
            print(afeyan_gt_dict_train[list(afeyan_gt_dict_train.keys())[3]].shape)

            original = afeyan_dict[list(afeyan_dict.keys())[3]].reshape(696 * 520, 31)
            gt_train = afeyan_gt_dict_train[list(afeyan_gt_dict_train.keys())[3]].reshape(696 * 520, 1)
            gt_test = afeyan_gt_dict_test[list(afeyan_gt_dict_test.keys())[3]].reshape(696 * 520, 1)

            gt = gt_train

            r = afeyan_dict[list(afeyan_dict.keys())[3]].shape[0]
            c = afeyan_dict[list(afeyan_dict.keys())[3]].shape[1]
            categories = 2
        #___________________________________________________________________________________________________________________________________________________

        if flag == 'indian':
            Ind_pines_dict = scio.loadmat('../../Dataset/Indian_pines.mat')
            Ind_pines_gt_dict = scio.loadmat('../../Dataset/Indian_pines_gt.mat')

            print(Ind_pines_dict['indian_pines'].shape)
            print(Ind_pines_gt_dict['indian_pines_gt'].shape)

            # remove the water absorption bands

            no_absorption = list(set(np.arange(0, 103)) | set(np.arange(108, 149)) | set(np.arange(163, 219)))

            original = Ind_pines_dict['indian_pines'][:, :, no_absorption].reshape(145 * 145, 200)

            print(original.shape)
            print('Remove wate absorption bands successfully!')

            gt = Ind_pines_gt_dict['indian_pines_gt'].reshape(145 * 145, 1)

            r = Ind_pines_dict['indian_pines'].shape[0]
            c = Ind_pines_dict['indian_pines'].shape[1]
            categories = 17
        if flag == 'pavia':
            pav_univ_dict = scio.loadmat('../../Dataset/PaviaU.mat')
            pav_univ_gt_dict = scio.loadmat('../../Dataset/PaviaU_gt.mat')

            print(pav_univ_dict['paviaU'].shape)
            print(pav_univ_gt_dict['paviaU_gt'].shape)

            original = pav_univ_dict['paviaU'].reshape(610 * 340, 103)
            gt = pav_univ_gt_dict['paviaU_gt'].reshape(610 * 340, 1)

            r = pav_univ_dict['paviaU'].shape[0]
            c = pav_univ_dict['paviaU'].shape[1]
            categories = 10

        if flag == 'houston':
            houst_dict = scio.loadmat('../../Dataset/Houston.mat')
            houst_gt_dict = scio.loadmat('../../Dataset/Houston_GT.mat')

            print(houst_dict['Houston'].shape)
            print(houst_gt_dict['Houston_GT'].shape)

            original = houst_dict['Houston'].reshape(349 * 1905, 144)
            gt = houst_gt_dict['Houston_GT'].reshape(349 * 1905, 1)

            r = houst_dict['Houston'].shape[0]
            c = houst_dict['Houston'].shape[1]
            categories = 16

        if flag == 'salina':
            salinas_dict = scio.loadmat('../../Dataset/Salinas_corrected.mat')
            salinas_gt_dict = scio.loadmat('../../Dataset/Salinas_gt.mat')

            print(salinas_dict['salinas_corrected'].shape)
            print(salinas_gt_dict['salinas_gt'].shape)

            original = salinas_dict['salinas_corrected'].reshape(512 * 217, 204)
            gt = salinas_gt_dict['salinas_gt'].reshape(512 * 217, 1)

            r = salinas_dict['salinas_corrected'].shape[0]
            c = salinas_dict['salinas_corrected'].shape[1]
            categories = 17

        if flag == 'ksc':
            ksc_dict = scio.loadmat('../../Dataset/KSC.mat')
            ksc_gt_dict = scio.loadmat('../../Dataset/KSC_gt.mat')

            print(ksc_dict['KSC'].shape)
            print(ksc_gt_dict['KSC_gt'].shape)

            original = ksc_dict['KSC'].reshape(512 * 614, 176)
            original[original > 400] = 0
            gt = ksc_gt_dict['KSC_gt'].reshape(512 * 614, 1)

            r = ksc_dict['KSC'].shape[0]
            c = ksc_dict['KSC'].shape[1]
            categories = 14

        rows = np.arange(gt_train.shape[0])  # start from 0
        
        # ID(row number), data, class number
        All_data = np.c_[rows, original, gt_train] # ID + DATA + gt (520 * 696, 1 + 31 + 1)

        # Removing background and obtain all labeled data
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # All ID of labeled  data
        
        


        return All_data, labeled_data, rows_num, categories, r, c, flag, gt_test


class product():
    def __init__(self, c, flag, All_data):
        self.c=c
        self.flag = flag
        self.All_data = All_data
    # product the training and testing pixel ID
    def generation_num(self, labeled_data, rows_num):
        samples_type = "fixed"#"fixed" # "fixed" # samples_type: ratio, fixed


        if samples_type == 'ratio':

            train_ratio = 0.8   

            train_num = []

            for i in np.unique(labeled_data[:, -1]):
                temp = labeled_data[labeled_data[:, -1] == i, :] # ground thruth ( labled data )
                temp_num = temp[:, 0]  # all ID of ground thruth ( labled data )
                
                

                #print(i, temp_num.shape[0])
                #np.random.seed(2020)
                np.random.shuffle(temp_num)  # random sequence
             
                if self.flag == 'afeyan':
    
                    split_th = 100
                    if len(temp) < split_th:
                        split_th =(int)(0.717391304347826 * len(temp)) # same stragy for others


                    train_num.append(temp_num[0:split_th])
                    
                if self.flag == 'indian':
                    if i == 1:
                        train_num.append(temp_num[0:33])
                    elif i == 7:
                        train_num.append(temp_num[0:20])
                    elif i == 9:
                        train_num.append(temp_num[0:14])
                    elif i == 16:
                        train_num.append(temp_num[0:75])
                    else:
                        train_num.append(temp_num[0:100])
                if self.flag == 'pavia' or self.flag=='houston' or self.flag=='salina':
                    train_num.append(temp_num[0:100])
                if self.flag == 'ksc':
                    if i==1:
                        train_num.append(temp_num[0:33])
                    elif i==2:
                        train_num.append(temp_num[0:23])
                    elif i==3:
                        train_num.append(temp_num[0:24])
                    elif i==4:
                        train_num.append(temp_num[0:24])
                    elif i==5:
                        train_num.append(temp_num[0:15])
                    elif i==6:
                        train_num.append(temp_num[0:22])
                    elif i==7:
                        train_num.append(temp_num[0:9])
                    elif i==8:
                        train_num.append(temp_num[0:38])
                    elif i==9:
                        train_num.append(temp_num[0:51])
                    elif i==10:
                        train_num.append(temp_num[0:39])
                    elif i==11:
                        train_num.append(temp_num[0:41])
                    elif i==12:
                        train_num.append(temp_num[0:49])
                    elif i==13:
                        train_num.append(temp_num[0:91])
                    #else:
                    #train_num.append(temp_num[0:int(temp.shape[0]*0.1)])

                trn_num = [x for j in train_num for x in j]  # merge
                np.random.shuffle(trn_num)
                val_num = trn_num[int(len(trn_num)*train_ratio):]
                tes_num = list(set(rows_num) - set(trn_num))
                pre_num = list(set(range(0, self.All_data.shape[0])) - set(trn_num))
                #trn_num = list(set(trn_num) | set(tes_num)) # for lichao mou's paper
                trn_num = trn_num[:int(len(trn_num)*train_ratio)]

    




        else:

            class_count = np.unique(labeled_data[:, -1]).shape[0] 

            split_size = 0.01
  
            train_num_expected = (int) (self.All_data.shape[0] * split_size// class_count)

            trn_num = []
            val_num = []
            tes_num = []
        
            sample_num = train_num_expected
            for i in range(class_count + 1):

                
                idx = self.All_data[self.All_data[:, -1] == i, 0]

                train_num_available = len(idx)

                np.random.shuffle(idx)
                
                if sample_num > train_num_available:
                   sample_num =  train_num_available


                else:
                    sample_num  = train_num_expected


                if i != 0: # labeled
                    trn_num.append(idx[:sample_num])
                    val_num.append(idx[sample_num : sample_num + class_count]) 
                    tes_num.append(idx[sample_num + class_count : ])
                
                else:
                    val_num.append(idx[:sample_num])
                    tes_num.append(idx[sample_num:])

            trn_num = [x for j in trn_num for x in j]  # merge
            val_num = [x for j in val_num for x in j]  # merge
            tes_num = [x for j in tes_num for x in j]  # merge
            pre_num = None # do not know

        print('number of training sample', len(trn_num ))
        print('number of validation sample', len(val_num ))
        print('number of testing sample', len(tes_num ))

        return rows_num, trn_num, val_num, tes_num, pre_num

                



    def production_label(self, num, y_map, split='Trn', gt_test = None):


        num = np.array(num)
        idx_2d = np.zeros([num.shape[0], 2]).astype(int)
        idx_2d[:, 0] = num // self.c
        idx_2d[:, 1] = num % self.c

        label_map = np.zeros(y_map.shape)
        
        for i in range(num.shape[0]):
            if split == 'Trn':
                label_map[idx_2d[i,0],idx_2d[i,1]] = self.All_data[num[i],-1]
            else:
                label_map[idx_2d[i,0],idx_2d[i,1]] = gt_test[num[i],-1]

        print('{} label map preparation Finished!'.format(split))
        return label_map

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]#output上分对的类别
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)#output上分对的类别中每类的个数
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)#output每类的个数
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)#target每类的个数
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()
