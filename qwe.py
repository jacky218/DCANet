import torch

checkpoint = torch.load("/home/honsen/tartan/TFNet-main/module_ori/bestMoudleNet_13.pth", map_location=torch.device('cpu'))

bestLoss = checkpoint['bestLoss']
bestLossEpoch = checkpoint['bestLossEpoch']
bestWerScore = checkpoint['bestWerScore']
bestWerScoreEpoch = checkpoint['bestWerScoreEpoch']
epoch = checkpoint['epoch']