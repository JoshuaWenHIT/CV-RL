#
#  detection.py
#  tools/datasets
#
#  Created by Joshua Wen on 2023/01/11.
#  Copyright © 2023 Joshua Wen. All rights reserved.
#
import os
from PIL import Image
from tools.utils.export import export_excel


class DetDatasetsStatistic:

    def __init__(self, dir_path):
        self.path = dir_path  # sequences dir
        self.format_list = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']
        self.sta_dict = dict()

    def main(self):
        for root, dirs, files in os.walk(self.path, topdown=True):
            if not files:
                self.sta_dict = {key: {'num': 0, 'res': None, 'format': ''} for key in dirs}
            else:
                key = root.split('/')[-1]
                self.get_num(key=key, length=len(files))
                img_first_path = os.path.join(root, files[0])
                self.get_res(key=key, img_path=img_first_path)
                self.get_format(key=key, img_name=files[0])
        return self.sta_dict

    def get_num(self, key, length):
        self.sta_dict[key]['num'] = length

    def get_res(self, key, img_path):
        img_size = Image.open(img_path).size
        self.sta_dict[key]['res'] = img_size

    def get_format(self, key, img_name):
        img_format = img_name.split('.')[-1]
        if img_format in self.format_list:
            self.sta_dict[key]['format'] = img_format
        else:
            print("{} is not an image !".format(img_name))


if __name__ == '__main__':
    DIR_PATH = "/media/joshuawen/Joshua_SSD3/Datasets/Det/VisDrone2019-VID-train/sequences"
    dataset = DetDatasetsStatistic(dir_path=DIR_PATH)
    sta = dataset.main()
    # export excel
    order = ['Seq.', 'Num.', 'Res.', 'Format']
    info = []
    for k in sta:
        info.append({'Seq.': k, 'Num.': '{}'.format(sta[k]['num']),
                     'Res.': '{}*{}'.format(sta[k]['res'][0], sta[k]['res'][1]),
                     'Format': sta[k]['format']})
    EXCEL_PATH = "/home/joshuawen/Documents/temp.xlsx"
    export_excel(info=info, output_path=EXCEL_PATH)
