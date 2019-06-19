import torch

class LossOnDict(torch.nn.Module):
    def __init__(self, key, loss):
        super(LossOnDict, self).__init__()
        self.key = key
        self.loss = loss
        
    def forward(self, pred_dict, label_dict):
        return self.loss(pred_dict[self.key], label_dict[self.key])

class PreApplyCriterionListDict(torch.nn.Module):
    """
    Wraps a loss operating on tensors into one that processes dict of labels and predictions
    """
    def __init__(self, criterions_single, sum_losses=True, loss_weights=None):
        super(PreApplyCriterionListDict, self).__init__()
        self.criterions_single = criterions_single
        self.sum_losses = sum_losses
        self.loss_weights = loss_weights

    def forward(self, pred_dict, label_dict):
        """
        The loss is computed as the sum of all the loss values
        :param pred_dict: List containing the predictions
        :param label_dict: List containing the labels
        :return: The sum of all the loss values computed
        """
        losslist = []
        for criterion_idx, criterion_single in enumerate(self.criterions_single):
            loss_i = criterion_single(pred_dict, label_dict)
            if self.loss_weights is not None:
                loss_i = loss_i * self.loss_weights[criterion_idx]
            losslist.append(loss_i)

        if self.sum_losses:
            return sum(losslist)
        else:
            return losslist

class AffineCropPositionPrior(torch.nn.Module):
    """
    """
    def __init__(self, fullFrameResolution, weight=0.1):
        super(AffineCropPositionPrior, self).__init__()
        self.key = 'spatial_transformer'
        self.weight = weight
        # without this aspect ratio the side that is longer in the image will also be longer in the crop
        # (becasue the x and y coordinates are normalized 0..1 irrespective of their true pixel length.
        # But we desire an equal aspect ratio:
        self.scale_aspectRatio = (torch.FloatTensor(fullFrameResolution)/min(fullFrameResolution)).cuda()
        self.scale_aspectRatio[0] *= 1.5 # makes x dimension smaller (1.5 as wide as y), to prefere tall crops for upright poses

    def forward(self, input_dict, label_dict_unused):
        affine_params = input_dict[self.key]
        scale_mean = 0.4
        trans_mean = 0
        diffs = 0

        translations = affine_params[:, :, :, 2]
        scales = torch.stack([affine_params[:, :, 0, 0],affine_params[:, :, 1,1]], dim=-1)

        # average position across batch (which is a sample of the whole dataset) should be the image center
        # take mean across batch (dim=1), number of transformers per image (dim=0) will be averaged after taking the difference.
        # Otherwise it is too easy to fulfill the prior with opposing positions and scales
        diffs += torch.mean((torch.mean(translations,dim=1)*self.scale_aspectRatio.unsqueeze(0).unsqueeze(0) - trans_mean)**2)
        # put a slight penality on scale, towards small scales
        diffs += torch.mean((torch.mean(scales,dim=1)*self.scale_aspectRatio.unsqueeze(0).unsqueeze(0) - scale_mean)**2)
        #diffs += torch.mean((torch.mean(scales[:, :, 1],dim=1)*self.scale_aspectRatio[1] - scale_mean)**2)
        return self.weight*diffs

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

class LossLabelMeanStdNormalized(torch.nn.Module):
    """
    Normalize the label before applying the specified loss (could be normalized loss..)
    """
    def __init__(self, key, loss_single, subjects=False, weight=1):
        super(LossLabelMeanStdNormalized, self).__init__()
        self.key = key
        self.loss_single = loss_single
        self.subjects = subjects
        self.weight=weight

    def forward(self, preds, labels):
        import IPython
        IPython.embed()
        pred_pose = preds[self.key]
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        label_pose_norm = (label_pose-label_mean)/label_std

        if self.subjects:
            info = labels['frame_info']
            subject = info.data.cpu()[:,3]
            errors = [self.loss_single.forward(pred_pose[i], label_pose_norm[i]) for i,x in enumerate(pred_pose) if subject[i] in self.subjects]
            #print('subject',subject,'errors',errors)
            if len(errors) == 0:
                return torch.autograd.Variable(torch.FloatTensor([0])).cuda()
            return self.weight * sum(errors) / len(errors)

        return self.weight * self.loss_single.forward(pred_pose,label_pose_norm)

class LossLabelMeanStdUnNormalized(torch.nn.Module):
    """
    UnNormalize the prediction before applying the specified loss (could be normalized loss..)
    """
    def __init__(self, key, loss_single, scale_normalized=False, weight=1):
        super(LossLabelMeanStdUnNormalized, self).__init__()
        self.key = key
        self.loss_single = loss_single
        self.scale_normalized = scale_normalized
        #self.subjects = subjects
        self.weight=weight

    def forward(self, preds, labels):
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        pred_pose = preds[self.key]
        
        if self.scale_normalized:
            IPython.embed() # TODO: normalize each pose
            per_frame_norm_label = label_pose.norm(dim=1, keepdim=True)
            per_frame_norm_pred  = pred_pose.norm(dim=1, keepdim=True)
            pred_pose = pred_pose / per_frame_norm_pred * per_frame_norm_label
        else:
            pred_pose_norm = (pred_pose*label_std) + label_mean
        return self.weight*torch.mean(self.loss_single.forward(pred_pose_norm, label_pose))