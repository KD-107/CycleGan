import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.transforms as Transforms
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import Dataset, dataloader
from config import *
from config import *
from dataset import HorseZebraDataset
from discriminator import Discriminator
from generator import Generator
from PIL import Image


class cycleGan():
    def __init__(self, pretrain=False):
        # 参数存放在config文件中
        self.lr = LEARNING_RATE
        # 批处理大小
        self.batch_size = BATCH_SIZE
        # 迭代次数
        self.epoch = EPOCH
        # horse图片保存的路径
        self.horse_root = HORSE_ROOT
        # zebra图片保存的路径
        self.zebra_root = ZEBRA_ROOT
        # 图片的通道数
        self.image_channels = IMAGE_CHANNELS

        # 权重保存的路径
        self.D_H_save_path = D_H_SAVE_PATH
        self.D_Z_save_path = D_Z_SAVE_PATH
        self.G_H_save_path = G_H_SAVE_PATH
        self.G_Z_save_path = G_Z_SAVE_PATH
        self.transforms = DataSetTransformes
        self.pretrain = pretrain

        self.device = DEVICE

    def train(self):
        D_H = Discriminator(in_channels=self.image_channels).to(self.device)
        D_Z = Discriminator(in_channels=self.image_channels).to(self.device)
        G_H = Generator(img_channels=self.image_channels).to(self.device)
        G_Z = Generator(img_channels=self.image_channels).to(self.device)
        if self.pretrain:
            # 加载horse判别器权重
            if os.path.exists(self.D_H_save_path):
                D_H.load_state_dict(torch.load(self.D_H_save_path))
                print(self.D_H_save_path + '权重加载完成')
            else:
                print(self.D_H_save_path + '权重加载失败')
            # 加载zebra判别器权重
            if os.path.exists(self.D_Z_save_path):
                D_Z.load_state_dict(torch.load(self.D_Z_save_path))
                print(self.D_Z_save_path + '权重加载完成')
            else:
                print(self.D_Z_save_path + '权重加载失败')
            # 加载horse生成器权重
            if os.path.exists(self.G_H_save_path):
                G_H.load_state_dict(torch.load(self.G_H_save_path))
                print(self.G_H_save_path + '权重加载完成')
            else:
                print(self.G_H_save_path + '权重加载失败')
            # 加载zebra生成器权重
            if os.path.exists(self.G_Z_save_path):
                G_Z.load_state_dict(torch.load(self.G_Z_save_path))
                print(self.G_Z_save_path + '权重加载完成')
            else:
                print(self.G_Z_save_path + '权重加载失败')

        # 损失函数
        MSE = nn.MSELoss()
        L1 = nn.L1Loss()

        # 优化器（两个判别器的参数一同更新）
        opt_D = torch.optim.Adam(params=list(D_H.parameters()) + list(D_Z.parameters()), lr=self.lr, betas=(0.5, 0.999))
        # （两个判别器的参数一同更新）
        opt_G = torch.optim.Adam(params=list(G_H.parameters()) + list(G_Z.parameters()), lr=self.lr, betas=(0.5, 0.999))
        dataset = HorseZebraDataset(horse_root=self.horse_root, zebra_root=self.zebra_root, transforms=self.transforms)
        mydataloader = DataLoader(dataset=dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        step = 1
        writer_horse_path = "fake_horse"
        writer_zebra_path = "fake_zebra"
        writer_horse = SummaryWriter(writer_horse_path)
        writer_zebra = SummaryWriter(writer_zebra_path)
        H_reals = 0
        H_fakes = 0
        for i in range(self.epoch):

            for index, data in enumerate(mydataloader, 1):
                print(index)
                horse_img, zebra_img = data
                horse_img = horse_img.to(self.device)
                zebra_img = zebra_img.to(self.device)
                # horse
                fake_horse = G_H(zebra_img)
                D_H_real = D_H(horse_img)
                D_H_fake = D_H(fake_horse.detach())
                H_reals += D_H_real.mean().item()
                H_fakes += D_H_fake.mean().item()

                D_H_real_loss = MSE(D_H_real, torch.ones_like(D_H_real))
                D_H_fake_loss = MSE(D_H_fake, torch.ones_like(D_H_fake))
                D_H_loss = D_H_real_loss + D_H_fake_loss

                # Zebra
                fake_zebra = G_Z(horse_img)
                D_Z_real = D_Z(zebra_img)
                D_Z_fake = D_Z(fake_zebra.detach())
                D_Z_real_loss = MSE(D_Z_real, torch.ones_like(D_Z_real))
                D_Z_fake_loss = MSE(D_Z_fake, torch.zeros_like(D_Z_fake))
                D_Z_loss = D_Z_real_loss + D_Z_fake_loss

                # 总损失
                D_loss = (D_H_loss + D_Z_loss) / 2
                opt_D.zero_grad()
                D_loss.backward()
                opt_D.step()

                # adversarial loss for both generators
                D_H_fake = D_H(fake_horse)
                D_Z_fake = D_Z(fake_zebra)
                loss_G_H = MSE(D_H_fake, torch.ones_like(D_H_fake))
                loss_G_Z = MSE(D_Z_fake, torch.zeros_like(D_Z_fake))

                # cycle loss
                cycle_zebra = G_Z(fake_horse)
                cycle_horse = G_H(fake_zebra)
                cycle_zebra_loss = L1(zebra_img, cycle_zebra)
                cycle_horse_loss = L1(horse_img, cycle_horse)

                # total loss
                G_loss = (
                        loss_G_Z
                        + loss_G_H
                        + 10 * cycle_horse_loss
                        + 10 * cycle_horse_loss
                )

                opt_G.zero_grad()
                G_loss.backward()
                opt_G.step()

                if index % 10 == 0:
                    with torch.no_grad():
                        D_H.eval()
                        D_Z.eval()
                        G_H.eval()
                        G_Z.eval()
                        image_grad_horse = torchvision.utils.make_grid(
                            fake_zebra, normalize=True
                        )
                        writer_zebra.add_image("fake_zebra", image_grad_horse, global_step=step)

                        step += 1
                        D_H.train()
                        D_Z.train()
                        G_H.train()
                        G_Z.train()

                print("[%d/epoch], H_reals: %f, H_fakesL %f" % (index, H_reals, H_fakes))

            self.save_weights(G_H, "epoch" + str(i) + '_G_H_' + str(H_reals) + "_" + str(H_fakes))
            self.save_weights(G_Z, "epoch" + str(i) + '_G_Z_' + str(H_reals) + "_" + str(H_fakes))
            self.save_weights(D_Z, "epoch" + str(i) + '_D_Z_' + str(H_reals) + "_" + str(H_fakes))
            self.save_weights(D_H, "epoch" + str(i) + '_D_H_' + str(H_reals) + "_" + str(H_fakes))


    # 生成horse风格或者zebra风格的特征图
    def GeneratorImg(self, zebra=True):
        generator = Discriminator(in_channels=self.image_channels)
        if zebra:
            if os.path.exists(self.G_Z_save_path):
                generator.load_state_dict(torch.load(self.G_Z_save_path))
        else:
            if os.path.exists(self.G_H_save_path):
                generator.load_state_dict(torch.load(self.G_H_save_path))
        while True:
            img_path = str(input())
            img = Image.open(img_path)
            img = self.transforms(img)
            img.unsqueeze_(dim=0)
            result_img = generator(img)
            result_img.squeeze_(dim=0)
            result_img = Transforms.ToPILImage()
            result_img.show()

    def save_weights(self, module, path):
        if os.path.exists(path):
            print(path + '文件已存在')
        else:
            torch.save(module.state_dict(), path)


if __name__ == '__main__':
    cycleGan = cycleGan()
    cycleGan.train()


