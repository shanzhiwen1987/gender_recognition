from torch import nn
from utils import Tester
from network import resnet34, resnet101
from torchvision import models
import torch
from torch import optim
import os
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F

def model(ckpt):

    gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test

    model = models.resnet101(pretrained=False)
    model.fc= nn.Linear(in_features=512*4, out_features=2)

    model.load_state_dict(torch.load(ckpt))
    gpu_test = str(gpus[0])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_test
    # Set Test parameters
    model = model.cuda()
    model.eval()

    return model



def process_img(img_dir):
    img = Image.open(img_dir)
    img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
    img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_input = Variable(torch.unsqueeze(img, 0))
    return img_input

model_body=model('./models/ckpt_body.pth')
model_face=model('./models/ckpt_face.pth')

img_list = os.listdir('./testimg/')
gpus=[0]
counter=0
with open('test_img.txt', 'r') as f:
    lines = f.readlines()

    for line in lines:
        item = line.strip().split(' ')


        #print('Processing image: ' + item[0])
        img_name=item[0]
        img_label=item[1]

        img_input=process_img(os.path.join('./body/', img_name))
        img_input = img_input.cuda()
        output = model_body(img_input)
        score_body = F.softmax(output, dim=1)


        img_input1=process_img(os.path.join('./face/',img_name))
        img_input1=img_input1.cuda()
        output1=model_face(img_input1)
        score_face=F.softmax(output1,dim=1)

        possibility=score_body[0][0].item()+score_face[0][0].item()
        if possibility>1:
            gender=0
        elif possibility<1:
            gender=1
        else:
            gender=2

        if gender==int(img_label):
            counter=counter+1
        else:print(img_name)
    print(counter/len(lines))


