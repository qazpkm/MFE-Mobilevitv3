from PIL import Image
import torchvision
import cv2
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from archs.mobilevit import MobileViT, InvertedResidual, MobileViT, ConvLayer
from archs.model_config import get_config
from archs.efficientnetv2 import EfficientNetV2
from torchvision import transforms
from models.spatial_transforms import *
from models.temporal_transforms import *


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.layers=nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                            nn.ReLU(inplace=True),
                            nn.LocalResponseNorm(2),
                            nn.MaxPool2d(kernel_size=3, stride=2))),
                ('features', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                  nn.ReLU(inplace=True))),
                # ('fc4', nn.Sequential(nn.Linear(500, 512),
                #                   nn.ReLU(inplace=True))),
                # ('fc5', nn.Sequential(nn.Dropout(0.5),
                #                   nn.Linear(500, 512),
                #                   nn.ReLU(inplace=True)))
        ]))

    def forward(self, x):
        avg_result = self.avgpool(x)
        output = self.layers(x)
        return output

def save_model(model):
    torch.save(obj=model, f='B.pth')

if __name__ == '__main__':
    
    # model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
    #                 [4, 3, 2, 4, 24, 48, 0, 0],
    #                 [4, 3, 2, 4, 48, 64, 0, 0],
    #                 [6, 3, 2, 4, 64, 128, 1, 0.25],
    #                 [9, 3, 1, 6, 128, 160, 1, 0.25],
    #                 [15, 3, 2, 6, 160, 256, 1, 0.25]]

    # emodel = EfficientNetV2(model_cnf=model_config,
    #                        num_classes=3,
    #                        dropout_rate=0)
    
    config = get_config("xx_small")
    base_model = MobileViT(config, num_classes=3)
    bmodel = MobileViT(config, num_classes=3)
    
    #bmodel = getattr(torchvision.models, 'resnet50')
    from models.action import Action
    for m in base_model.modules():
        if isinstance(m, InvertedResidual):
            for nmnm in m.block:
                if nmnm == m.block[0]:
                    if isinstance(nmnm, ConvLayer)and len(nmnm.block) == 3 and m.use_res_connect:
                        nmnm.block[0] = Action(nmnm.block[0], n_segment=3, shift_div=8)
        else:
            asdasda = 1
            # print('error')
    net = base_model
    save_model(net)
    # 图片路径
    img_path = 'data/img1.jpg'
    img_size = 224
    # 给图片进行标准化操作
    img = Image.open(img_path).convert('RGB')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(img_size * 1.143)),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
    # input_mean = [.485, .456, .406]
    # input_std = [.229, .224, .225]
    # normalize = GroupNormalize(input_mean, input_std)
    # scales = [1, .875, .75, .66]
    # transforms = torchvision.transforms.Compose([
    #         GroupScale([224, 224]),
    #         GroupMultiScaleCrop([224, 224], scales),
    #         Stack(roll=(base_model in ['BNInception', 'InceptionV3'])),
    #         ToTorchFormatTensor(div=(base_model not in ['BNInception', 'InceptionV3'])),
    #         normalize])
    # temporal_transform_train = torchvision.transforms.Compose([
    #         TemporalUniformCrop_train(args.clip_len)
    #     ])
    
    data = transforms(img).unsqueeze(0)
    # 用于加载Pycharm中封装好的网络框架
    # model = torchvision.models.vgg11_bn(pretrained=True)
    # 用于加载1中生成的.pth文件
    model = torch.load(f="B.pth")
    # 打印一下刚刚生成的.pth文件看看他的网络结构
    print(model)
    model.eval()
    #实例化
    #net = MDNet()
    #save_model(net)
    #print('23132131231')
    #print(base_model.forward)
    features=net.forward123(data)
    print(features)
    # model = torch.load(f="A.pth")
    features.retain_grad()
    # t = model.avgpool(features)
    # t = t.reshape(1, -1)
    # output = model.classifier(t)[0]
    # pred = torch.argmax(output).item()
    # pred_class = output[pred]
    #
    # pred_class.backward()
    grads = features.grad

    features = features[0]
    # avg_grads = torch.mean(grads[0], dim=(1, 2))
    # avg_grads = avg_grads.expand(features.shape[1], features.shape[2], features.shape[0]).permute(2, 0, 1)
    # features *= avg_grads

    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    cv2.imshow('1', superimposed_img)
    cv2.waitKey(0)
