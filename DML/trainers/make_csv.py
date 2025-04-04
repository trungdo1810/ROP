# Create a csv file contains the path of all images and their corresponding labels
# The csv file will be used to train the model

import os
import pandas as pd
import numpy as np

root = '../datasets'
class_list = ['normal', 'preplus', 'plus']

def make_csv():
    data = []
    for i, cls in enumerate(class_list):
        class_path = os.path.join(root, cls)
        for img in os.listdir(class_path):
            img_path = os.path.join(cls, img)
            data.append([img_path, i])
    data = np.array(data)
    df = pd.DataFrame(data, columns=['path', 'label'])
    df.to_csv('../datasets/data.csv', index=False)

if __name__ == '__main__':
    print("Creating csv file...")
    make_csv()

