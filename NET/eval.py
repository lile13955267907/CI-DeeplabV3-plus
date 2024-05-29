import os
import sys
import torch
import cv2
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
from utils import decode_segmap
from SegDataFolder import semData
from getSetting import get_yaml, get_criterion, get_optim, get_scheduler, get_net
from metric import AverageMeter, intersectionAndUnion
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


# for transform
# 需要计算文件夹对应图片的mean&std，或者使用trainset的mean std
# mean = np.array([0.03369877519145767, 0.02724876914897716, 0.019425700166401196, 0.03361996144023702,
#                       0.032213421442935264, 0.03168874880981447, 0.031527547610634275])       # 训练原图的平均值
# std = np.array([0.0033888922358425873, 0.0038459901628366544, 0.004287149017973415, 0.01173405280221371,
#                       0.3572610932095403, 0.3572878634587769, 0.3572964354695759])      # 训练原图的方差"""
# mean = np.array([0.03464819234212239, 0.02909945320250497, 0.021297139243474077, 0.03537811477903337, 0.022892621043371777,
# 0.02226419697110616, 0.022086084614102802, 0.042612235121942094])
#
# std = np.array([ 0.004163198603400601, 0.004996242018132609, 0.005034741271796365, 0.011400495383521463, 0.25227939359138707,
# 0.2523133853625742, 0.2523220137758833, 0.027253946192470957])

mean = np.array([0.032566757489748646, 0.02637095626112056, 0.018314198199367013, 0.03309229557376886, 0.017596356968105787,
0.016523874620367734, 0.016279129986987696, 0.1212707391848252])

std = np.array([ 0.0032750218477141106, 0.004101055859099182, 0.004508486721736283, 0.010343705582897232, 0.14504942060838555,
0.14507033497806596, 0.14507565500017827, 0.12671439102702925])
# mean = np.array([0.03154230127334593,  0.016846031904220588, 0.018314198199367013, 0.03196724393367766, 0.014989601993560779,
# # 0.013698559713363647, 0.013421019506454458, 0.15998547165327537])
# #
# # std = np.array([ 0.0029905543103276194, 0.0036020510088444783, 0.004508486721736283, 0.009767130059320967, 0.007060040565573836,
# # 0.006838838114168965, 0.006805446993368705, 0.1382348987427858])
def get_idx(channels):
    assert channels in [2, 4, 7, 8, 3]
    if channels == 3:
        return list(range(3))
    elif channels == 4:
        return list(range(4))
    elif channels == 7:
        return list(range(7))
    elif channels == 8:
        return list(range(8))
    elif channels == 2:
        return list(range(6))[-2:]

def getTransorm4eval(channel_idx=[0,1,2,3]):
    import transform as T
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=mean[channel_idx],std=std[channel_idx])
        ]
    )
def convert_to_color(image, output_path):
    # 创建一个空的彩色图像
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # 将三个值分别转换为不同的颜色
    # 假设值是0, 1, 2
    # 0 -> 蓝色 (255, 0, 0)
    # 1 -> 绿色 (0, 255, 0)
    # 2 -> 红色 (0, 0, 255)
    color_image[np.where(image == 0)] = [0, 0, 0]
    color_image[np.where(image == 1)] = [0, 255, 0]
    color_image[np.where(image == 2)] = [0, 0, 255]

    # 保存彩色图像
    cv2.imwrite(output_path, color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def eval(args,config,):
    os.makedirs(args.outputdir, exist_ok=True)

    net = get_net(config) 
    softmax = nn.Softmax(dim=1)
    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    net = net.cuda() if args.use_gpu else net

    dataset = semData(
        train=False,
        root='./Data',
        channels = config['channels'],
        transform=getTransorm4eval(channel_idx=get_idx(config['channels'])),
        selftest_dir=args.testdir
    )
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)
    
    net.eval()
    with torch.no_grad():
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        correct = []
        preds = []



        for i, batch in enumerate(dataloader):
            img = batch['X'].cuda() if args.use_gpu else batch['X']
            label = batch['Y'].cuda() if args.use_gpu else batch['Y']
            path = batch['path'][0].split('.')[0]
            # label[label == 3] = 0

            outp = net(img)
            
            score = softmax(outp)      
            pred = score.max(1)[1]
            # pred[pred == 3] = 0

            saved_1 = pred.squeeze().cpu().numpy()
            saved_255 = 122 * saved_1

            # 保存预测图片，像素值为0&1
            cv2.imwrite(r'D:\BARELAND\NET\eval_output\eval_output_1/{}.png'.format(path), saved_1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # 保存预测图片，像素值为0&255
            cv2.imwrite(r'D:\BARELAND\NET\eval_output\eval_output_255/{}.png'.format(path), saved_255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # 将三值图像转换为彩色图像并保存
            convert_to_color(saved_1, r'D:\BARELAND\NET\eval_output\eval_ouput_color/{}.tif'.format(path))
            
            correct.extend(label.reshape(-1).cpu().numpy())
            preds.extend(pred.reshape(-1).cpu().numpy())
            
            N = img.size()[0]
            
            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), label.cpu().numpy(), config['n_classes'], config['ignore_index'])
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            acc = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        print(iou_class)
        acc_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(acc_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum)+1e-10)
        precision = precision_score(correct, preds, average='micro')
        recall = recall_score(correct, preds, average='micro')
        f1 = f1_score(correct, preds, average='micro')
        f1_scores_per_class = f1_score(correct, preds, average=None)
        for i, score in enumerate(f1_scores_per_class):
            print(f"Class {i} F1 Score: {score}")


        print(acc_class)
        print('mIoU {:.4f} | mAcc {:.4f} | allAcc {:.4f} | precision {:.4f} | recall {:.4f} | f1 {:.4f}'.format(
            mIoU, mAcc, allAcc, precision, recall, f1))
        # print("小麦总产量：2688449斤 、油菜总产量：150391斤")
        

### argument parse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--testdir',type=str,default='eval', help='path of test images') # 测试文件夹，需要放在Data文件夹下，跟train,test有相同结构
parser.add_argument('--outputdir',type=str,default='eval_output',help='test output') # 指定生成预测图片的输出文件夹
parser.add_argument('--gpu',type=int, default=0,help='gpu index to run') # gpu卡号，单卡默认为0
parser.add_argument('--configfile',type=str, default='ConfigFiles/config-deeplab.yaml',help='path to config files') # 选定configfile,用于指定网络
parser.add_argument('--checkpoint',type=str, default=r"D:\BARELAND\NET\deeplab+_3Adam_gaijin_lr=0.0001_3batch\DeepLab-0.8270-ep7.pth", help='checkpoint path') # 加载定模型路径
args = parser.parse_args()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if torch.cuda.is_available():
        args.use_gpu = True
    else:
        args.use_gpu = False
    
    config = get_yaml(args.configfile)
    eval(args, config)


