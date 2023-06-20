import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import transforms

"""
=============================自己的包===========================
"""
from data_processing.BraTS2020 import *
from data_processing.BraTS2019 import *
from networks.MyNet.MyNet import *
# from networks import Unet
from utils import *


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data info
    data_path = "../dataset/brats2019/data"
    test_txt = "../dataset/brats2019/train.txt"
    patch_size = (240, 240, 128)
    test_set = BraTS2019(data_path, test_txt, pre=True, transform=transforms.Compose([
        # RandomRotFlip(),
        CenterCrop(patch_size),
        ToTensor()
    ]))

    print("using {} device.".format(device))

    # 1-坏疽(NT,necrotic tumor core),2-浮肿区域(ED,peritumoral edema),4-增强肿瘤区域(ET,enhancing tumor)
    # 评价指标：ET(label4),TC(label1+label4),WT(label1+label2+label4)
    print(f"using {MODEL_NAME} for prediction")
    if MODEL_NAME == 'MyNet':
        model = MyNet(in_channels=4, num_classes=4).to(device)

    # 加载训练模型
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        print('Successfully loading checkpoint.')

    model.eval()
    # (86,45) (36,30) (28,45) 45,40 -- train
    d1 = test_set[86]
    slice_w = 50
    image, label = d1
    print("image.shape:", image.shape)
    print("label.shape:", label.shape)

    inputs_wt = image[0].unsqueeze(0).unsqueeze(0)
    print(inputs_wt.shape)
    inputs_tc = (image[1]+image[2]).unsqueeze(0).unsqueeze(0)
    print(inputs_tc.shape)
    inputs_et = image[3].unsqueeze(0).unsqueeze(0)
    print(inputs_et.shape)

    yuzhi_wt, yuzhi_tc, yuzhi_et = 0.9, 0.9, 0.9
    output_wt = model(inputs_wt.to(device)).squeeze(0)
    print("max:output", torch.max(output_wt))
    pre_wt = torch.sigmoid(output_wt)
    print("max:pre", torch.max(pre_wt))
    pre_wt = (pre_wt > yuzhi_wt).float()
    print("torch.unique(pre):", torch.unique(pre_wt))
    pre_wt = pre_wt.squeeze(0).detach().cpu().numpy()
    print("pre.shape:", pre_wt.shape)
    print("loss:", F.binary_cross_entropy(torch.from_numpy(pre_wt).to(torch.float32), label.to(torch.float32)))

    output_tc = model(inputs_tc.to(device)).squeeze(0)
    pre_tc = torch.sigmoid(output_tc)
    pre_tc = (pre_tc > yuzhi_tc).float()
    pre_tc = pre_tc.squeeze(0).detach().cpu().numpy()

    output_et = model(inputs_et.to(device)).squeeze(0)
    pre_et = torch.sigmoid(output_et)
    pre_et = (pre_et > yuzhi_et).float()
    pre_et = pre_et.squeeze(0).detach().cpu().numpy()

    f, axarr = plt.subplots(3, 4, figsize=(10, 7))

    # flair
    axarr[0][0].title.set_text('Flair')
    axarr[0][0].imshow(image[0, :, :, slice_w], cmap="gray")
    axarr[0][0].axis('off')

    # t1ce
    axarr[0][1].title.set_text('T1ce')
    axarr[0][1].imshow(image[1, :, :, slice_w], cmap="gray")
    axarr[0][1].axis('off')

    # t1
    axarr[0][2].title.set_text('T1')
    axarr[0][2].imshow(image[2, :, :, slice_w], cmap="gray")
    axarr[0][2].axis('off')

    # t2
    axarr[0][3].title.set_text('T2')
    axarr[0][3].imshow(image[3, :, :, slice_w], cmap="gray")
    axarr[0][3].axis('off')

    # WT
    mask_segmentation_wt = label[:, :, slice_w].clone()
    mask_segmentation_wt[mask_segmentation_wt == 0] = 0
    mask_segmentation_wt[mask_segmentation_wt == 1] = 1
    mask_segmentation_wt[mask_segmentation_wt == 2] = 1
    mask_segmentation_wt[mask_segmentation_wt == 3] = 1

    axarr[1][0].imshow(mask_segmentation_wt, cmap='gray')
    axarr[1][0].title.set_text('GT WT')
    axarr[1][0].axis('off')

    # pre WT
    mask_segmentation_wt = pre_wt[:, :, slice_w]
    axarr[1][1].imshow(mask_segmentation_wt.astype('uint8'), cmap='gray')
    axarr[1][1].title.set_text('pre WT')
    axarr[1][1].axis('off')

    # TC
    mask_segmentation_tc = label[:, :, slice_w].clone()
    mask_segmentation_tc[mask_segmentation_tc == 0] = 0
    mask_segmentation_tc[mask_segmentation_tc == 1] = 1
    mask_segmentation_tc[mask_segmentation_tc == 2] = 0
    mask_segmentation_tc[mask_segmentation_tc == 3] = 1

    axarr[1][2].imshow(mask_segmentation_tc, cmap='gray')
    axarr[1][2].title.set_text('GT TC')
    axarr[1][2].axis('off')

    # pre TC
    mask_segmentation_tc = pre_tc[:, :, slice_w]
    axarr[1][3].imshow(mask_segmentation_tc.astype('uint8'), cmap='gray')
    axarr[1][3].title.set_text('pre TC')
    axarr[1][3].axis('off')

    # ET
    mask_segmentation_et = label[:, :, slice_w].clone()
    mask_segmentation_et[mask_segmentation_et == 0] = 0
    mask_segmentation_et[mask_segmentation_et == 1] = 0
    mask_segmentation_et[mask_segmentation_et == 2] = 0
    mask_segmentation_et[mask_segmentation_et == 3] = 1

    axarr[2][0].imshow(mask_segmentation_et, cmap='gray')
    axarr[2][0].title.set_text('GT ET')
    axarr[2][0].axis('off')

    # pre ET
    mask_segmentation_et = pre_et[:, :, slice_w]
    axarr[2][1].imshow(mask_segmentation_et.astype('uint8'), cmap='gray')
    axarr[2][1].title.set_text('pre ET')
    axarr[2][1].axis('off')

    # WT 1,2,4
    mask_segmentation = label[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 0, 0]
    color_segmentation[mask_segmentation == 2] = [0, 255, 0]
    color_segmentation[mask_segmentation == 3] = [0, 0, 255]
    axarr[2][2].imshow(color_segmentation.astype('uint8'), cmap="gray")
    axarr[2][2].title.set_text('GT All')
    axarr[2][2].axis('off')

    # pre all
    output = pre_wt
    output[output == 1] = 2

    # 还原label 1
    temp = pre_tc
    output[temp == 1] = 1

    # 还原label 3
    temp = pre_et
    output[temp == 1] = 3
    print(output.shape)
    # pre WT
    mask_segmentation = output[:, :, slice_w]
    print(np.max(mask_segmentation))
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 0, 0]
    color_segmentation[mask_segmentation == 2] = [0, 255, 0]
    color_segmentation[mask_segmentation == 3] = [0, 0, 255]
    axarr[2][3].imshow(color_segmentation.astype('uint8'), cmap="gray")
    axarr[2][3].title.set_text('pre ALL')
    axarr[2][3].axis('off')

    # # GT TC 1,4
    # mask_segmentation = label[:, :, slice_w]
    # color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    # color_segmentation[mask_segmentation == 1] = [255, 255, 255]
    # color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    # axarr[2][0].imshow(color_segmentation.astype('uint8'), cmap="gray")
    # axarr[2][0].title.set_text('GT TC')
    # axarr[2][0].axis('off')
    #
    # # pre TC
    # mask_segmentation = pre[:, :, slice_w]
    # color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    # color_segmentation[mask_segmentation == 1] = [255, 255, 255]
    # color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    # axarr[2][1].imshow(color_segmentation.astype('uint8'), cmap="gray")
    # axarr[2][1].title.set_text('pre TC')
    # axarr[2][1].axis('off')
    #
    # # GT ET 4
    # mask_segmentation = label[:, :, slice_w]
    # color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    # color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    # axarr[2][2].imshow(color_segmentation.astype('uint8'), cmap="gray")
    # axarr[2][2].title.set_text('GT ET')
    # axarr[2][2].axis('off')
    #
    # # pre ET
    # mask_segmentation = pre[:, :, slice_w]
    # color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    # color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    # axarr[2][3].imshow(color_segmentation.astype('uint8'), cmap="gray")
    # axarr[2][3].title.set_text('pre ET')
    # axarr[2][3].axis('off')

    plt.show()


if __name__ == "__main__":
    MODEL_NAME = 'MyNet'
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--data_path', type=str, default='../dataset/brats2020/data')
    parser.add_argument('--train_txt', type=str, default='../dataset/brats2020/train.txt')
    parser.add_argument('--valid_txt', type=str, default='../dataset/brats2020/valid.txt')
    parser.add_argument('--train_log', type=str, default=f'./results/{MODEL_NAME}/{MODEL_NAME}.txt')
    parser.add_argument('--weights', type=str, default=f'./results/{MODEL_NAME}/{MODEL_NAME}.pth')
    parser.add_argument('--save_path', type=str, default=f'./checkpoint/{MODEL_NAME}')

    args = parser.parse_args()

    main(args)



