import torch
import torchvision.transforms as Transforms


# 学习率
LEARNING_RATE = 2e-4
# 批处理尺寸
BATCH_SIZE = 4
# 迭代次数
EPOCH = 150
# 相互转换类的图片存放路径
HORSE_ROOT = "./horse2zebra/horse2zebra/trainB"
ZEBRA_ROOT = "./horse2zebra/horse2zebra/trainB"
# 图片的通道数
IMAGE_CHANNELS = 3
# 判别器权重存放的路径
D_H_SAVE_PATH = '\logs\D_H'
D_Z_SAVE_PATH = '\logs\D_Z'
# 生成器权重存放的路径
G_H_SAVE_PATH = '\logs\G_H'
G_Z_SAVE_PATH = '\logs\G_Z'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DataSetTransformes = Transforms.Compose(
    [
        Transforms.Resize(256),
        Transforms.RandomVerticalFlip(p=0.5),
        Transforms.ToTensor()
    ]
)