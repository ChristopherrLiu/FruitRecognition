# -*- coding:utf-8 -*-
import os
import cv2
import json
import random
import matplotlib.pyplot as plt
import os.path as osp

BASE_DIR = osp.dirname(osp.abspath(__file__))

if __name__ == '__main__' :
    data_dir = osp.join(BASE_DIR, "FIDS30")
    all_label = os.listdir(data_dir)
    all_label = [label for label in all_label if len(label.split('.')) <= 1]
    id2label = {key : value for key, value in enumerate(all_label)}
    label2id = {key : value for value , key in enumerate(all_label)}
    json.dump(id2label, open(osp.join(BASE_DIR, "labels.json"), "w"))

    train_counter, test_counter = 0, 0
    train_img_info, test_img_info = dict(), dict()
    fruit_num, fruit_name = list(), list()
    for fruit_type in all_label :
        cur_img_dir = osp.join(data_dir, fruit_type)
        img_ids = os.listdir(cur_img_dir)
        fruit_num.append(len(img_ids))
        fruit_name.append(fruit_type)
        test_lists = random.sample(img_ids, 5)
        for img_id in img_ids :
            cur_img_path = osp.join(cur_img_dir, img_id)
            try :
                img = cv2.imread(cur_img_path)
                new_img_path = osp.join(BASE_DIR, "dataset", str(label2id[fruit_type]) + '_' + img_id)
                cv2.imwrite(new_img_path, img)
                if img_id in test_lists :
                    test_img_info[str(test_counter)] = {
                        "filename" : str(label2id[fruit_type]) + '_' + img_id,
                        "path" : new_img_path,
                        "label" : fruit_type
                    }
                    test_counter = test_counter + 1
                else :
                    train_img_info[str(train_counter)] = {
                        "filename": str(label2id[fruit_type]) + '_' + img_id,
                        "path": new_img_path,
                        "label": fruit_type
                    }
                    train_counter = train_counter + 1
            except Exception as e:
                print(e)
                print(fruit_type, img_id)
    json.dump(train_img_info, open(osp.join(BASE_DIR, "train_imgs.json"), "w"))
    json.dump(test_img_info, open(osp.join(BASE_DIR, "test_imgs.json"), "w"))
    plt.bar(range(len(fruit_name)), fruit_num)
    plt.xticks(range(len(fruit_name)), fruit_name, rotation=65)
    plt.legend()
    plt.show()