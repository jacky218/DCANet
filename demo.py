import Net
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from WER import WerScore
import os
import DataProcessMoudle
import videoAugmentation
import numpy as np
import decode
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from evaluation import evaluteMode
from evaluationT import evaluteModeT
import random
import cv2
import csv
import decode
from DataProcessMoudle import PreWords
import ssl

import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
PAD = ' '

device = 'cuda'
fmap_block = list()
def forward_hook(module, input, output):
    fmap_block.append(output)       #N, C, T, H, ,W

def hook_model(model):
        model.conv2d.corr1.corrAttn.register_forward_hook(forward_hook)

def cam_show_img(img, feature_map, grads, out_dir):  # img: ntchw, feature_map: ncthw, grads: ncthw
    # N, C, T, H, W = feature_map.shape

    T, C, H, W = feature_map.shape
    cam = np.zeros((T,H,W), dtype=np.float32)	# thw

    # grads = grads[0,:].reshape([C, T, -1])

    grads = grads.reshape([T, C, -1])
    weights = np.mean(grads, axis=-1)

    for i in range(C):
        for j in range(T):
            # cam[j] += weights[i,j] * feature_map[0, i, j, :, :]
            cam[j] += weights[j, i] * feature_map[ j, i, :, :]
    cam = np.maximum(cam, 0)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        import shutil
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    for i in range(T):
        out_cam = cam[i]
        out_cam = out_cam - np.min(out_cam)
        out_cam = out_cam / (1e-7 + out_cam.max())
        out_cam = cv2.resize(out_cam, (img.shape[3], img.shape[4]))
        out_cam = (255 * out_cam).astype(np.uint8)
        heatmap = cv2.applyColorMap(out_cam, cv2.COLORMAP_JET)
        cam_img = np.float32(heatmap) / 255 + (img[0,i]/2+0.5).permute(1,2,0).cpu().data.numpy()
        cam_img = cam_img/np.max(cam_img)
        cam_img = np.uint8(255 * cam_img)
        path_cam_img = os.path.join(out_dir, f"cam_{i}.jpg")
        cv2.imwrite(path_cam_img, cam_img)
    print('Generate cam.jpg')

def pca_analysis(img):
    from sklearn.decomposition import PCA

    MC1 = np.load("left.npy")
    MC2 = np.load("right.npy")

    T = MC1.shape[0]

    for i in range(T):
        mc1 = MC1[i]
        mc2 = MC2[i]

        pca2 = PCA(n_components=1)
        pca2.fit(mc1)
        mc1 = pca2.transform(mc1)

        pca = PCA(n_components=1)
        pca.fit(mc2)
        mc2 = pca.transform(mc2)

        mc1 = mc1.reshape(28,28)
        mc2 = mc2.reshape(28, 28)

        out_mc1 = cv2.resize(mc1, (img.shape[3], img.shape[4]))
        out_mc2 = cv2.resize(mc2, (img.shape[3], img.shape[4]))

        out_mc1 = (20 * out_mc1).astype(np.uint8)
        out_mc2 = (20 * out_mc2).astype(np.uint8)

        heatmap1 = cv2.applyColorMap(out_mc1, cv2.COLORMAP_JET)
        cam_img1 = np.float32(heatmap1) / 255 + (img[0, i] / 2 + 0.5).permute(1, 2, 0).cpu().data.numpy()
        cam_img1 = cam_img1 / np.max(cam_img1)
        cam_img1 = np.uint8(255 * cam_img1)
        path_cam_img1 = os.path.join("/home/honsen/tartan/TFNet-main/ORI_visual/pca_left", f"pca_{i}.jpg")
        cv2.imwrite(path_cam_img1, cam_img1)

        heatmap2 = cv2.applyColorMap(out_mc2, cv2.COLORMAP_JET)
        cam_img2 = np.float32(heatmap2) / 255 + (img[0, i] / 2 + 0.5).permute(1, 2, 0).cpu().data.numpy()
        cam_img2 = cam_img2 / np.max(cam_img2)
        cam_img2 = np.uint8(255 * cam_img2)
        path_cam_img2 = os.path.join("/home/honsen/tartan/TFNet-main/ORI_visual/pca_right", f"pca_{i}.jpg")
        cv2.imwrite(path_cam_img2, cam_img2)

def Word2Id(trainLabelPath,validLabelPath,testLabelPath, dataSetName):

    if dataSetName == "CE-CSL":
        wordList = []

        wordList = []
        with open(trainLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    words = row[3].split("/")
                    words = PreWords(words)
                    wordList += words

        with open(validLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    words = row[3].split("/")
                    words = PreWords(words)
                    wordList += words

        with open(testLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    words = row[3].split("/")
                    words = PreWords(words)
                    wordList += words

    idx2word = [PAD]
    set2list = sorted(list(set(wordList)))
    idx2word.extend(set2list)

    word2idx = {w: i for i, w in enumerate(idx2word)}

    return word2idx, len(idx2word) - 1, idx2word

def inference(modulepath):

    transformTest = videoAugmentation.Compose([
        videoAugmentation.CenterCrop(224),
        videoAugmentation.ToTensor(),
    ])

    word2idx, wordSetNum, idx2word = Word2Id( "/home/honsen/tartan/CE-CSL/CE-CSL/label/train.csv","/home/honsen/tartan/CE-CSL/CE-CSL/label/dev.csv","/home/honsen/tartan/CE-CSL/CE-CSL/label/test.csv",
                                                               "CE-CSL")

    decoder = decode.Decode(word2idx, wordSetNum + 1, 'beam')

    validData = DataProcessMoudle.MyDataset("/home/honsen/tartan/CE-CSL/CE-CSL/video/demo", "/home/honsen/tartan/CE-CSL/CE-CSL/label/demo.csv", word2idx, "CE-CSL",
                                            transform=transformTest)

    validLoader = DataLoader(dataset=validData, batch_size=1, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)

    moduleNet = Net.moduleNet(1024, wordSetNum * 1 + 1, 'CorrNet', device, "CE-CSL", True)
    moduleNet = moduleNet.to(device)

    checkpoint = torch.load(modulepath, map_location=torch.device('cuda'))
    moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])

    moduleNet.train()
    print("开始验证模型")
    # 验证模型
    werScoreSum = 0
    loss_value = []
    total_info = []
    total_sent = []

    hook_model(moduleNet)

    logSoftMax = nn.LogSoftmax(dim=-1)
    count = 0
    for Dict in tqdm(validLoader):

        if count <260:
            count+=1
            continue

        data = Dict["video"].to(device)

        # pca_analysis(data)

        label = Dict["label"]
        dataLen = Dict["videoLength"]
        info = Dict["info"]
        ##########################################################################
        targetOutData = [torch.tensor(yi).to(device) for yi in label]
        targetLengths = torch.tensor(list(map(len, targetOutData)))
        targetData = targetOutData
        targetOutData = torch.cat(targetOutData, dim=0).to(device)
        batchSize = len(targetLengths)


        logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, False)

        logProbs1 = logSoftMax(logProbs1)


        grads_val_l = torch.load('/home/honsen/tartan/TFNet-main/left.pth').cpu().data.numpy()
        grads_val_r = torch.load('/home/honsen/tartan/TFNet-main/right.pth').cpu().data.numpy()

        fmap = fmap_block[0].cpu().data.numpy()

        cam_show_img(data,fmap,grads_val_l, out_dir='/home/honsen/tartan/TFNet-main/CAM_left')
        cam_show_img(data, fmap, grads_val_r, out_dir='/home/honsen/tartan/TFNet-main/CAM_right')

        pred, targetOutDataCTC = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

        predictionResultStr = [idx2word[j] for j in targetOutDataCTC]

        gtResultStr = [idx2word[j] for j in targetOutData]

        print(predictionResultStr)
        print(gtResultStr)

        print("============")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    modulepath = "/home/honsen/tartan/TFNet-main/module/bestMoudleNet_48.pth"
    inference(modulepath)
