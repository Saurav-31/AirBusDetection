import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = '/mnt/sqnap/ntarigo/datasets/kaggle/ship/'
masks = pd.read_csv(os.path.join(root_dir, 'train_ship_segmentations.csv'))

not_empty = pd.notna(masks.EncodedPixels)
print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')
print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')
masks.head()

print("EmptyImages", (~not_empty).sum()*100/masks.ImageId.nunique(), '%')

masks[not_empty].EncodedPixels.apply(lambda x: np.array(x.split(" ")[1:][::2], dtype=np.int).sum())
plt.show()

masks[not_empty].EncodedPixels.apply(lambda x: len(np.array(x.split(" ")[1:][::2], dtype=np.int))).hist()
plt.show()


masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
df = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()

sns.distplot(df.ships, bins=5)
plt.show()

df.ships.value_counts()

imgnames = df[df.ships > 14]['ImageId'].values
from PIL import Image


for i in range(20):
    img = Image.open(os.path.join(root_dir,'train/', imgnames[i]))
    img = img.convert('RGB')
    plt.subplot(5, 4, i+1)
    plt.imshow(img)

plt.show()


