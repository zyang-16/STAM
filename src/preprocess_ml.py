import numpy as np
from collections import defaultdict
import os
import math
import random

file_name = "../data/ml-1m/ratings.dat"
write_file = "../data/ml-1m/ratings.txt"

user_map = {}
item_map = {}

def read_ml(file_name):
    user_items_times = defaultdict(list)
    with open(file_name, 'r') as f:
        for line in f:
            user_id, item_id, ratings, timestamp = line.strip().split('::')
            user_items_times[user_id].append((item_id, timestamp, ratings))

    flag = True
    # filter feedback less than 5
    while flag:
        user_items_times_filter = defaultdict(list)
        for user in list(user_items_times.keys()):
            if len(user_items_times[user]) < 5:
                pass
            else:
                user_items_times_filter[user].extend(user_items_times[user])

        item_dict = defaultdict(list)
        for user in list(user_items_times_filter.keys()):
            for item, time, ratings in user_items_times_filter[user]:
                item_dict[item].append((user, time, ratings))

        item_user_time_filter = defaultdict(list)
        for item in list(item_dict.keys()):
            if len(item_dict[item]) < 5:
                pass
            else:
                item_user_time_filter[item].extend(item_dict[item])

    
        user_items_times = defaultdict(list)
        for item in list(item_user_time_filter.keys()):
            for user, time, ratings in item_user_time_filter[item]:
                user_items_times[user].append((item, time, ratings))
        
        iter = 0
        for key in user_items_times:
            if len(user_items_times[key]) < 5:
                iter += 1
                break

        if iter == 0:
            flag = False
        else:
            flag = True

    user_item_time_sorted = sorted(user_items_times.items(), key=lambda x: int(x[0]))
    
    # change user_id and item_id
    user_items_times_ok = defaultdict(list)
    user_id = 0
    for user, item_list in user_item_time_sorted:
        user_map[user] = user_id
        user_items_times_ok[user_id].extend(item_list)
        user_id += 1

    item_id = user_id
    item_set = list(item_user_time_filter.keys())
    for item in item_set:
        item_map[item] = item_id
        item_id += 1


    with open(write_file, 'w') as T:
        for key in user_items_times_ok:
            # sorted by timestamp
            sorted_user_bh = sorted(user_items_times_ok[key], key=lambda x:x[1])
            print(sorted_user_bh)
            for item, time, ratings in sorted_user_bh:
                T.write(str(key) + '\t' + str(item_map[item]) + '\t' + ratings + '\n')

    with open("../data/ml-1m/ratings_ok.txt", 'w') as T:
        for key in user_items_times_ok:
            sorted_user_bh = sorted(user_items_times_ok[key], key=lambda x:x[1])
            for item, time, ratings in sorted_user_bh:
                T.write(str(key) + '\t' + str(item_map[item]) + '\t' + ratings + '\t' + str(time) + '\n')





def data_split(path, filename, total_num, split_num):
    user_bh = defaultdict(list)
    user_bh_test = defaultdict(list)
    user_bh_train = defaultdict(list)
    fr = open(filename, 'r')
    for line in fr:
        user, item, rating = line.strip().split('\t')
        user_bh[user].append(item)
    
    random.seed(10)
    num = random.random()
    for key in user_bh:
        length = len(user_bh[key])
        test_length = length * 1.0/total_num*split_num
        if math.modf(test_length)[0] > num:
            test_length = int(math.modf(test_length)[1]) + 1
        else:
            test_length = int(math.modf(test_length)[1])
        train_length = length - test_length
        user_bh_train[key].extend(user_bh[key][0: train_length])
        user_bh_test[key].extend(user_bh[key][train_length:])

    if not os.path.exists(path):
        os.makedirs(path)
    f_train = open(os.path.join(path,'train.txt'), 'w')
    f_test = open(os.path.join(path, 'test.txt'), 'w')
    for key in user_bh_train:
        for item in user_bh_train[key]:
            f_train.write(str(key) + '\t' + str(item) + '\t' + '1' + '\n')
    for key in user_bh_test:
        for item in user_bh_test[key]:
            f_test.write(str(key) + '\t' + str(item) + '\t' + '1' + '\n')

        

read_ml(file_name)
data_split('../data/ml-1m/test/', write_file, 10, 2)
data_split('../data/ml-1m/validation/', '../data/ml-1m/test/train.txt', 8, 1)
