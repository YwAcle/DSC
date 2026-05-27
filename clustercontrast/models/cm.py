import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.float().mm(ctx.features.float()).half()
            # grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = self.features[y]/self.features[y].norm()

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss




#实例级
class EM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update, not applied for meta learning
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def em(inputs, indexes, features, momentum=0.5):
    return EM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class Memory_wise_v3(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v3, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        # labels--(each src and predicted tgt id and outliers), 13638


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = self.features[y]/self.features[y].norm()
        # del inputs,indexes

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)
        mask_instance = self.compute_mask(sim.size(), indexes,device=sim.device)
        sim_exp_intra = sim_exp #* mask_intra
        # print('sim_exp_intra',sim_exp_intra)
        score_intra =   F.softmax(sim_exp_intra,dim=1)#sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # print('score_intra',score_intra)
        score_intra = score_intra.clamp_min(1e-8)
        ins_loss = -score_intra.masked_select(mask_instance.bool()).log().mean()
        return ins_loss#* 0.6
    def compute_mask(self, size, img_ids,device=None):
        # mask_inter = torch.ones(size, device=device)
        # for i, cam in enumerate(cam_ids):
        #     intra_cam_ids = self.cam2uid[cam]
        #     mask_inter[i, intra_cam_ids] = 0

        # mask_intra = 1 - mask_inter
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance#, mask_intra, mask_inter


    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.ones(size, device=device)#zeros
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers

    

class EnhancedClusterMemory(nn.Module):
    def __init__(self, num_features, num_samples, classifier, classifier_cue=None, 
                 temp=0.05, momentum=0.2, use_hard=False, classifier_weight=0.1):
        super(EnhancedClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.classifier = classifier  # 主分类器
        self.classifier_cue = classifier_cue  # 辅助分类器
        self.classifier_weight = classifier_weight  # 分类器kl损失权重
        
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        
    def forward(self, inputs, targets):
        inputs_norm = F.normalize(inputs, dim=1).cuda()
        
        # 标准对比学习部分
        if self.use_hard:
            outputs = cm_hard(inputs_norm, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs_norm, targets, self.features, self.momentum)
        
        outputs /= self.temp
        contrastive_loss = F.cross_entropy(outputs, targets)
        
        # 分类器对齐部分，确保分类器输出对齐
        if self.classifier is not None and self.classifier_cue is not None:
            
            # 分割特征向量
            feat_1, feat_2 = torch.split(inputs_norm, self.num_features//2, dim=1)
            feat_1 = F.normalize(feat_1, dim=1)
            feat_2 = F.normalize(feat_2, dim=1)


            # 当前特征的分类结果
            logits_main, _ = self.classifier(feat_1)
            logits_cue, _ = self.classifier_cue(feat_2)

            feat_1 = feat_1*8
            feat_2 = feat_2*8
            
            # 使用相同的温度系数
            prob_main = torch.softmax(logits_main, dim=1)
            prob_cue = torch.softmax(logits_cue, dim=1)

            prob_main = torch.cat([prob_main,prob_cue],dmi=1)
            
            # 获取目标中心并计算其分类结果
            target_centers = self.features[targets]  # 已归一化的聚类中心
            center_1, center_2 = torch.split(target_centers, self.num_features//2, dim=1)
            center_1 = F.normalize(center_1, dim=1)
            center_2 = F.normalize(center_2, dim=1)
            
            target_logits_main, _ = self.classifier(center_1)
            target_logits_cue, _ = self.classifier_cue(center_2)

            center_1 = center_1*8
            center_2 = center_2*8
            
            target_prob_main = torch.softmax(target_logits_main, dim=1)
            target_prob_cue = torch.softmax(target_logits_cue, dim=1)

            target_prob_main = torch.cat([target_prob_main,target_prob_cue],dim=1)
            
            # 使用KL散度计算概率分布间的差异
            def kl_loss(pred, target):
                # 添加小的epsilon防止log(0)
                epsilon = 1e-8
                pred = torch.clamp(pred, epsilon, 1.0) 
                return F.kl_div(pred.log(), target, reduction='batchmean')
            
            classifier_loss = kl_loss(prob_main, target_prob_main.detach())
            
            # 总损失
            loss = contrastive_loss + self.classifier_weight * classifier_loss
        else:
            loss = contrastive_loss
        
        return loss
