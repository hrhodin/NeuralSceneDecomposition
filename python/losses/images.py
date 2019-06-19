import torch
import numpy as np
from models import resnet_low_level

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class SobelCriterium(torch.nn.Module):
    """
    Approximates horizontal and vertical gradients with the Sobel operator and puts a criterion on these gradient estimates.
    """
    def __init__(self, criterion, weight=1):
        super(SobelCriterium, self).__init__()
        self.weight = weight
        self.criterion = criterion

        kernel_x = np.array([[1, 0, -1], [2,0,-2],  [1, 0,-1]])
        kernel_y = np.array([[1, 2,  1], [0,0, 0], [-1,-2,-1]])

        channels = 3
        kernel_size = 3
        self.conv_x = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.conv_x.weight = torch.nn.Parameter(torch.from_numpy(kernel_x).float().unsqueeze(0).unsqueeze(0).expand([channels,channels,kernel_size,kernel_size]))
        self.conv_x.weight.requires_grad = False
        self.conv_x.cuda()
        self.conv_y = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = torch.nn.Parameter(torch.from_numpy(kernel_y).float().unsqueeze(0).unsqueeze(0).expand([channels,channels,kernel_size,kernel_size]))
        self.conv_y.weight.requires_grad = False
        self.conv_y.cuda()
        
    def forward(self, pred, label):
        pred_x = self.conv_x.forward(pred)
        pred_y = self.conv_y(pred)
        label_x = self.conv_x(label)
        label_y = self.conv_y(label)

        return self.weight * (self.criterion(pred_x, label_x) + self.criterion(pred_y, label_y))

class ImageNetCriterium(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """
    def __init__(self, criterion, weight=1, do_maxpooling=True):
        super(ImageNetCriterium, self).__init__()
        self.weight = weight
        self.criterion = criterion

        self.net = resnet_low_level.resnet18(pretrained=True, num_channels = 3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.to(device)
        
    def forward(self, pred, label):
        preds_x  = self.net(pred)
        labels_x = self.net(label)
        
        losses = [self.criterion(p, l) for p,l in zip(preds_x,labels_x)]

        return self.weight * sum(losses) / len(losses)

class LossInstanceMeanStdFromLabel(torch.nn.Module):
    """
    Normalize the pose before applying the specified loss (done per batch element)
    """
    def __init__(self, loss_single):
        super(LossInstanceMeanStdFromLabel, self).__init__()
        self.loss_single = loss_single

    def forward(self, preds, labels):
        pred_pose = preds
        label_pose = labels
        batch_size = label_pose.shape[0]
        feature_size = label_pose.shape[1]
        eps = 0.00001 # to prevent division by 0
        # build mean and std across third and fourth dimension and restore afterwards again
        label_mean = torch.mean(label_pose.view([batch_size,feature_size,-1]),dim=2,keepdim=False).view([batch_size,-1,1,1])
        pose_mean  = torch.mean(pred_pose.view( [batch_size,feature_size,-1]),dim=2,keepdim=False).view([batch_size,-1,1,1])
        label_std  = torch.std( label_pose.view([batch_size,feature_size,-1]),dim=2,keepdim=False).view([batch_size,-1,1,1]) + eps
        pose_std   = torch.std( pred_pose.view( [batch_size,feature_size,-1]),dim=2,keepdim=False).view([batch_size,-1,1,1]) + eps

        pred_pose_norm = ((pred_pose - pose_mean)/pose_std)*label_std + label_mean
        if torch.isnan(pred_pose_norm).any():
            print('LossInstanceMeanStdFromLabel: torch.isnan(pred_pose_norm)')
            IPython.embed()

        return self.loss_single.forward(pred_pose_norm,label_pose)
