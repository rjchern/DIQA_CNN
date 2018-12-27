import pandas as pd
import numpy as np
import yaml

class DataInfoLoader:
    def __init__(self,dataset_name,config):
        self.config=config
        self.dataset_name=dataset_name
        self.img_num=len(pd.read_excel(config[dataset_name]['gt_file_path'])['img_name'])
        self.IQA_results_path=config[dataset_name]['IQA_results_path']
    
    def get_qs_std(self):
        """Standard ground truth quality score of all images
        
        Returns:
        - pandas.Series store the GT for all images
        """
        return pd.read_excel(self.config[self.dataset_name]['gt_file_path'])['acc_avg']

    def get_img_name(self):
        return pd.read_excel(self.config[self.dataset_name]['gt_file_path'])['img_name']
    
    def get_img_set(self):
        return pd.read_excel(self.config[self.dataset_name]['gt_file_path'])['img_set']

    def get_img_path(self):
        "Get the image path for all images"
        return [self.config[self.dataset_name]['root']+'/'+name for name in self.get_img_name()]

if __name__=='__main__':
    with open('./config.yaml') as f:
        config=yaml.load(f)
    dataset_name='SOC'
    dil=DataInfoLoader(dataset_name,config)
    img_name=dil.get_img_name()
    print(img_name[2])
