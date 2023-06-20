import os
import argparse

import numpy as np
import torch
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
=============================自己的包===========================
"""
from data_processing.BraTS2020 import *
from data_processing.BraTS2019 import *
from networks.MyNet.MyNet import *
from networks.MyNet.baseline import *
from networks.MyNet.baseline_gn import *
from networks.MyNet.baseline_attention import *
from networks.MyNet.baseline_extract import *

from utils import *


def val_loop(model, criterion, loader, device):
    model.eval()
    running_loss = 0
    dice1_val = 0
    dice2_val = 0
    dice3_val = 0
    sen_WT, sen_ET, sen_TC = 0, 0, 0
    spe_WT, spe_ET, spe_TC = 0, 0, 0
    ds_wt, ds_et, ds_tc = 0, 0, 0

    pbar = tqdm(loader, desc='Validation: ')
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = torch.softmax(outputs,dim=1)

            loss = criterion(outputs, masks)
            dice1, dice2, dice3 = cal_dice(outputs, masks)

            running_loss += loss.item()
            dice1_val += dice1.item()
            dice2_val += dice2.item()
            dice3_val += dice3.item()

            # with open(f"{MODEL_NAME}_et.txt", 'a+') as f:
            #     f.write("{:.3f}, ".format(dice1.item()))
            #     if step % 20 == 0:
            #         f.write('\n')
            # with open(f"{MODEL_NAME}_wt.txt", 'a+') as f:
            #     f.write("{:.3f}, ".format(dice3.item()))
            #     if step % 20 == 0:
            #         f.write('\n')
            # with open(f"{MODEL_NAME}_tc.txt", 'a+') as f:
            #     f.write("{:.3f}, ".format(dice2.item()))
            #     if step % 20 == 0:
            #         f.write('\n')

            pbar.set_postfix(loss=f"{loss:.3f}", dice1=f'{dice1:.3f}', dice2=f"{dice2:.3f}", dice3=f"{dice3:.3f}")

            # oh_label = F.one_hot(masks, 4).permute(0, 4, 1, 2, 3).to(device)
            # oh_output = torch.sigmoid(outputs).to(device)
            oh_label = F.one_hot(masks, 4).permute(0, 4, 1, 2, 3).detach().cpu().numpy()
            oh_output = torch.argmax(outputs, dim=1).long()
            oh_output = F.one_hot(oh_output, 4).permute(0, 4, 1, 2, 3).detach().cpu().numpy()

            sen_WT += sensitivity_WT(oh_output, oh_label)
            sen_ET += sensitivity_ET(oh_output, oh_label)
            sen_TC += sensitivity_TC(oh_output, oh_label)
            spe_WT += specificity_WT(oh_output, oh_label)
            spe_ET += specificity_ET(oh_output, oh_label)
            spe_TC += specificity_TC(oh_output, oh_label)
            ds_wt += hausdorff_distance_WT(torch.from_numpy(oh_output), torch.from_numpy(oh_label))
            ds_et += hausdorff_distance_ET(torch.from_numpy(oh_output), torch.from_numpy(oh_label))
            ds_tc += hausdorff_distance_TC(torch.from_numpy(oh_output), torch.from_numpy(oh_label))

    sen_WT = sen_WT / len(loader)
    sen_ET = sen_ET / len(loader)
    sen_TC = sen_TC / len(loader)
    spe_WT = spe_WT / len(loader)
    spe_ET = spe_ET / len(loader)
    spe_TC = spe_TC / len(loader)
    ds_wt = ds_wt / len(loader)
    ds_et = ds_et / len(loader)
    ds_tc = ds_tc / len(loader)

    loss = running_loss / len(loader)
    dice1 = dice1_val / len(loader)
    dice2 = dice2_val / len(loader)
    dice3 = dice3_val / len(loader)
    return {'loss': loss, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3,
            'sen_WT': sen_WT, 'sen_ET': sen_ET, 'sen_TC': sen_TC,
            'spe_WT': spe_WT, 'spe_ET': spe_ET, 'spe_TC': spe_TC,
            'ds_wt': ds_wt, 'ds_et': ds_et, 'ds_tc': ds_tc}


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data info
    size = (128, 128, 128)
    val_dataset = BraTS2019(args.data_path, args.train_txt, pre=True, transform=transforms.Compose([
        CenterCrop(size),
        ToTensor()
    ]))
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False,
                            pin_memory=True)

    print("using {} device.".format(device))
    print("using {} images for validation.".format(len(val_dataset)))
    # img,label = train_dataset[0]

    # 1-坏疽(NT,necrotic tumor core),2-浮肿区域(ED,peritumoral edema),4-增强肿瘤区域(ET,enhancing tumor)
    # 评价指标：ET(label4),TC(label1+label4),WT(label1+label2+label4)
    print(f"using {MODEL_NAME} for Evaluation.")
    if MODEL_NAME == 'MyNet':
        model = MyNet(in_channels=4, num_classes=4).to(device)
    elif MODEL_NAME == 'baseline':
        model = Baseline(in_channels=4, num_classes=4).to(device)
    elif MODEL_NAME == 'baseline_gn':
        model = Baseline_gn(in_channels=4, num_classes=4).to(device)
    elif MODEL_NAME == 'baseline_attention':
        model = Baseline_attention(in_channels=4, num_classes=4).to(device)
    elif MODEL_NAME == 'baseline_extract':
        model = Baseline_extract(in_channels=4, num_classes=4).to(device)

    criterion = Loss(n_classes=4, weight=torch.tensor([0.25, 0.25, 0.25, 0.25])).to(device)

    # 加载训练模型
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        print('Successfully loading checkpoint.')

    metrics2 = val_loop(model, criterion, val_loader, device)
    # metrics3 = val_loop(model, criterion, test_loader, device)

    print("Valid -- loss: {:.5f} ET: {:.5f} TC: {:.5f} WT: {:.5f}".format(metrics2['loss'], metrics2['dice1'],
                                                                          metrics2['dice2'], metrics2['dice3']))
    print("Valid -- sen_WT: {:.5f} sen_ET: {:.5f} sen_TC: {:.5f}".format(metrics2['sen_WT'], metrics2['sen_ET'],
                                                                         metrics2['sen_TC']))
    print("Valid -- spe_WT: {:.5f} spe_ET: {:.5f} spe_TC: {:.5f}".format(metrics2['spe_WT'], metrics2['spe_ET'],
                                                                         metrics2['spe_TC']))
    print("Valid -- ds_wt: {:.3f} ds_et: {:.3f} ds_tc: {:.3f}".format(metrics2['ds_wt'], metrics2['ds_et'],
                                                                      metrics2['ds_tc']))

    # print("Test  -- loss: {:.5f} ET: {:.5f} TC: {:.5f} WT: {:.5f}".format(metrics3['loss'], metrics3['dice1'],
    #                                                                       metrics3['dice2'], metrics3['dice3']))
    # print("Test -- sen_WT: {:.5f} sen_ET: {:.5f} sen_TC: {:.5f}".format(metrics3['sen_WT'], metrics3['sen_ET'],
    #                                                                     metrics3['sen_TC']))
    # print("Test -- spe_WT: {:.5f} spe_ET: {:.5f} spe_TC: {:.5f}".format(metrics3['spe_WT'], metrics3['spe_ET'],
    #                                                                     metrics3['spe_TC']))
    # print("Test -- ds_wt: {:.3f} ds_et: {:.3f} ds_tc: {:.3f}".format(metrics3['ds_wt'], metrics3['ds_et'],
    #                                                                  metrics3['ds_tc']))


if __name__ == '__main__':
    MODEL_NAME = 'MyNet'
    # MODEL_NAME = 'baseline'
    # MODEL_NAME = 'baseline_gn'
    # MODEL_NAME = 'baseline_extract'
    # MODEL_NAME = 'baseline_attention'

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--data_path', type=str, default='../dataset/brats2019/data')
    parser.add_argument('--train_txt', type=str, default='../dataset/brats2019/train.txt')
    parser.add_argument('--valid_txt', type=str, default='../dataset/brats2020/valid.txt')
    parser.add_argument('--train_log', type=str, default=f'./results/{MODEL_NAME}/{MODEL_NAME}.txt')
    parser.add_argument('--weights', type=str, default=f'./results/{MODEL_NAME}/{MODEL_NAME}.pth')
    parser.add_argument('--save_path', type=str, default=f'./checkpoint/{MODEL_NAME}')

    args = parser.parse_args()

    main(args)

