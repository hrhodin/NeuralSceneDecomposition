import torch
from IPython import embed as Ie

class Criterion3DPose_leastSquaresScaled(torch.nn.Module):
    """
    Normalize the scale in the least squares sense, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_leastSquaresScaled, self).__init__()
        self.criterion = criterion

    def forward(self, pred, label):
        #Optimal scale transform
        batch_size = pred.size()[0]
        ST_size = pred.size()[1]
        pred_vec = pred.view(batch_size,ST_size,-1)
        gt_vec = label.view(batch_size,ST_size,-1)
        dot_pose_pose = torch.sum(torch.mul(pred_vec,pred_vec), dim=-1, keepdim=True)
        dot_pose_gt   = torch.sum(torch.mul(pred_vec,gt_vec), dim=-1, keepdim=True)

        s_opt = dot_pose_gt / dot_pose_pose

        return self.criterion.forward(s_opt.expand_as(pred)*pred, label)

class MPJPECriterion(torch.nn.Module):
    """
    Mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, weight=1, reduction='elementwise_mean'):
        super(MPJPECriterion, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, label):
        size_orig = pred.size()
        batchSize = size_orig[0]
        diff = pred.view(batchSize,-1) - label.view(batchSize,-1)
        diff_sq = torch.mul(diff,diff)

        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)

        #print('diff_3d_len_sq', diff_3d_len_sq.size())
        if self.reduction == 'sum':
            return self.weight*torch.sum(diff_3d_len);    # mean across batch and joints
        elif self.reduction == 'none':
            return self.weight*diff_3d_len;    # mean across batch and joints
        else: #if self.reduction == 'elementwise_mean':
            return self.weight*torch.mean(diff_3d_len);    # mean across batch and joints

class StaticDenormalizedLoss(torch.nn.Module):
    """
    Denormalize by std and mean before loss computation. Should improve output statistics to be unit variance, but not alter loss
    """
    def __init__(self, key, loss_single):
        super(StaticDenormalizedLoss, self).__init__()
        self.key = key
        self.loss_single = loss_single

    def forward(self, preds, labels):
        pred_pose_norm = preds[self.key]
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        pred_pose = pred_pose_norm*label_std + label_mean
        return self.loss_single.forward(pred_pose, label_pose)