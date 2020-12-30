import random
import numpy as np
import heapq
import scipy.io
import math

class data_loader:
    def __init__(self, database_name='UCB', labeled_num=100):
        if database_name == 'UWA' or database_name == 'UWA30':
            self.class_num = 30
            self.depth_feature_length = 110
            self.rgb_feature_length = 3 * 2048
        elif database_name == 'UCB':
            self.class_num = 11
            self.depth_feature_length = 110
            self.rgb_feature_length = 3 * 2048
        elif database_name == 'DHA':
            self.class_num = 23
            self.depth_feature_length = 110
            self.rgb_feature_length = 3 * 2048
        elif database_name == 'nyu' or database_name == 'NYUD' or database_name == 'nyu_resnet18':
            self.class_num = 10
            self.depth_feature_length = 1024
            self.rgb_feature_length = 1024
        self.filename = database_name
        self.labeled_num = int(labeled_num)
        self.data_x = []
        self.data_label = []
        self.train_data_x = []
        self.train_data_y = []
        self.train_data_label = []
        self.test_data_x = []
        self.test_data_y = []
        self.test_data_label = []
        self.train_data_xy = []
        self.test_data_xy = []
        
        # average
        self.r_ave = []
        self.d_ave = []

        self.rgb_neighboor = []
        self.depth_neighboor = []

    def read_train_action(self, p=-1):

        self.data_x = []
        self.data_y = []
        self.data_label = []
        self.train_data_x = []
        self.train_data_y = []
        self.train_data_label = []
        self.test_data_x = []
        self.test_data_y = []
        self.test_data_label = []
        self.only_x = []
        self.only_x_l = []
        self.only_y = []
        self.only_y_l = []
        # p = 0.3
        feature_num1 = 110
        feature_num2 = 3*2048
        feature_num = feature_num1 + feature_num2
        num = 0
        f = open('new_data/' + self.filename+'_total_train.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            a = random.random()
            if (a<p/2):
                self.only_x.append(row[0:feature_num1])
                self.only_x_l.append(row[feature_num1+feature_num2:])
            elif (a<p):
                self.only_y.append(row[feature_num1:feature_num1 + feature_num2])
                self.only_y_l.append(row[feature_num1+feature_num2:])
            else:
                self.data_x.append(row[0:feature_num1])
                self.data_y.append(row[feature_num1:feature_num1+feature_num2])
                self.data_label.append(row[feature_num:])
                self.train_data_x.append(row[0:feature_num1])
                self.train_data_y.append(row[feature_num1:feature_num1+feature_num2])
                self.train_data_xy.append(row[0:feature_num1 + feature_num2])
                self.train_data_label.append(row[feature_num1+feature_num2:])
        f.close()
        f = open('new_data/' + self.filename+'_total_test.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            self.data_x.append(row[0:feature_num1])
            self.data_y.append(row[feature_num1:feature_num1+feature_num2])
            self.data_label.append(row[feature_num:])
            self.test_data_x.append(row[0:feature_num1])
            self.test_data_y.append(row[feature_num1:feature_num1 + feature_num2])
            self.test_data_xy.append(row[0:feature_num1 + feature_num2])
            self.test_data_label.append(row[feature_num1 + feature_num2:])
        f.close()
        self.sample_total_num = len(self.data_x)
        self.sample_train_num = len(self.train_data_x)
        self.sample_test_num = len(self.test_data_x)
        self.sample_only_x_num = len(self.only_x)
        self.sample_only_y_num = len(self.only_y)
            
        print(self.sample_total_num)

        # split two list for edge loss
        self.same_label_list = []
        self.diff_label_list = []
        for i in range(self.sample_train_num -1):
            for j in range(i+1, self.sample_train_num):
                if (self.train_data_label[i] == self.train_data_label[j]):
                    self.same_label_list.append((i,j))
                else:
                    self.diff_label_list.append((i,j))
        self.same_label_num = len(self.same_label_list)
        self.diff_label_num = len(self.diff_label_list)

    
    def read_train_image(self, p=-1):

        self.data_x = []
        self.data_y = []
        self.data_label = []
        self.train_data_x = []
        self.train_data_y = []
        self.train_data_label = []
        self.test_data_x = []
        self.test_data_y = []
        self.test_data_label = []
        self.only_x = []
        self.only_x_l = []
        self.only_y = []
        self.only_y_l = []
        if self.filename == 'nyu':
            feature_num1 = 1024  # depth
            feature_num2 = 1024  # rgb
            feature_num = feature_num1 + feature_num2
            num = 0
            addr = './nyu_feature.mat'
            feature = scipy.io.loadmat(addr)
        elif self.filename == 'nyu_resnet18':
            feature_num1 = 512  # depth
            feature_num2 = 512  # rgb
            feature_num = feature_num1 + feature_num2
            num = 0
            addr = './nyu_feature_resnet18.mat'
            feature = scipy.io.loadmat(addr)
        elif self.filename == 'SUN':
            feature_num1 = 512  # depth
            feature_num2 = 512  # rgb
            feature_num = feature_num1 + feature_num2
            num = 0
            addr = './SUN_feature_resnet18.mat'
            feature = scipy.io.loadmat(addr)
      
        self.data_x = np.concatenate((feature['train_dep'], feature['test_dep']), axis=0)
        self.data_y = np.concatenate((feature['train_rgb'], feature['test_rgb']), axis=0)
        self.data_label = np.concatenate((feature['train_label'], feature['test_label']), axis=0)
        self.train_data_x = feature['train_dep']
        self.train_data_y = feature['train_rgb']
        self.train_data_label = feature['train_label']
        
        self.test_data_x = feature['test_dep']
        self.test_data_y = feature['test_rgb']
        self.test_data_label = feature['test_label']

        self.sample_total_num = len(self.data_x)
        self.sample_train_num = len(self.train_data_x)
        self.sample_test_num = len(self.test_data_x)
        self.sample_only_x_num = len(self.only_x)
        self.sample_only_y_num = len(self.only_y)
            
        print(self.sample_total_num)

       

    def train_next_batch_RGBDepth(self, _batch_size):
        xx = []
        yy = []
        zz = []
        for sample_num in random.sample(range(self.sample_train_num), _batch_size):
            xx.append(self.train_data_xy[sample_num])
            zz.append(self.train_data_label[sample_num])
        return yy, xx, zz

    def test_next_batch_RGBDepth(self, _batch_size):
        xx = []
        yy = []
        zz = []
        for sample_num in random.sample(range(self.sample_test_num), _batch_size):
            xx.append(self.test_data_xy[sample_num])
            zz.append(self.test_data_label[sample_num])
        return yy, xx, zz

    def train_next_batch(self, _batch_size):
        xx = []
        yy = []
        zz = []
        for sample_num in random.sample(range(self.sample_train_num), _batch_size):
            xx.append(self.train_data_x[sample_num])
            yy.append(self.train_data_y[sample_num])
            zz.append(self.train_data_label[sample_num])
        return yy, xx, zz
    def train_only_depth_next_batch(self, _batch_size):
        xx = []
        zz = []
        for sample_num in random.sample(range(self.sample_only_x_num), _batch_size):
            xx.append(self.only_x[sample_num])
            zz.append(self.only_x_l[sample_num])
        return xx, zz

    def train_only_rgb_next_batch(self, _batch_size):
        yy = []
        zz = []
        for sample_num in random.sample(range(self.sample_only_y_num), _batch_size):
            yy.append(self.only_y[sample_num])
            zz.append(self.only_y_l[sample_num])
        return yy, zz

    def test_next_batch(self, _batch_size):
        xx = []
        yy = []
        zz = []
        for sample_num in random.sample(range(self.sample_test_num), _batch_size):
            xx.append(self.test_data_x[sample_num])
            yy.append(self.test_data_y[sample_num])
            zz.append(self.test_data_label[sample_num])
        return yy, xx, zz

    def DA_init(self):
        return self.train_data_y, self.test_data_y, self.train_data_x, self.test_data_x, self.train_data_label, self.test_data_label

    def DA_train_next_batch(self, _batch_size = 100):
        xx = []
        yy = []
        zz = []
        pp = []
        qq = []
        num = int(self.sample_train_num * self.labeled_num / 100.0)
        train_list = list(range(num)) * int(math.ceil(_batch_size / num))
        train_idx = random.sample(train_list, _batch_size)
        for sample_num in train_idx:
            xx.append(self.train_data_x[sample_num])
            yy.append(self.train_data_y[sample_num])
            zz.append(self.train_data_label[sample_num])
        for sample_num in random.sample(range(self.sample_test_num), _batch_size):
            pp.append(self.test_data_x[sample_num])
            qq.append(self.test_data_y[sample_num])
            
        return yy, xx, zz, qq, pp
