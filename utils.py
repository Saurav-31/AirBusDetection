import numpy as np # linear algebra
from skimage.morphology import label
import matplotlib.pyplot as plt

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:, :] += scale(i) * rle_decode(mask)
    return all_masks

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_sample(inp, masks):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    inp = cnv2numpy(inp)
    masks = cnv2numpy(masks)
    ax1.imshow(inp)
    ax2.imshow(masks)
    plt.show()

def show_sample2(inp, masks, gt, imgname):
    # print(inp.size(), masks.size(), gt.size())
    # print(inp.max(), masks.max(), gt.max())
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    masks = masks.cpu().detach().numpy().transpose((1, 2, 0))
    gt = gt.cpu().detach().numpy().transpose((1, 2, 0))
    ax1.imshow(inp)
    ax2.imshow(masks)
    ax3.imshow(gt)
    #plt.show()
    fig.savefig("image_dump/{}".format(imgname))

def show_sample3(inp, masks, imgname):
    # print(inp.size(), masks.size())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    masks = masks.cpu().detach().numpy().transpose((1, 2, 0))
    ax1.imshow(inp)
    ax2.imshow(masks)
    plt.show()
    fig.savefig("test_image_dump/{}".format(imgname))

def cnv2numpy(x):
    x = x.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = std * x + mean
    x = np.clip(x, 0, 1)
    return x