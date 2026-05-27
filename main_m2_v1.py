import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import ast
import time
from datetime import timedelta
from sklearn.cluster import DBSCAN
import sys
import torch
import torch.nn.functional as F
from collections import Counter
from torch.backends import cudnn
import random
import argparse
import numpy as np
from torch.cuda import amp
from data_loader.loader import Loader
from torch.utils.data import DataLoader
from core import Base, train, test,test_vcm,test_bupt
from tools import make_dirs, os_walk, time_now,Logger
import warnings
from torchvision.transforms import InterpolationMode
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import VideoPreprocessor, VideoPreprocessor_color
import os.path as osp
from clustercontrast import datasets
from clustercontrast.evaluators import Evaluator, extract_features
import collections
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
import torch.nn.functional as F
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from clustercontrast.utils.data import IterLoader
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, ChannelExchange
from clustercontrast.models.cm import ClusterMemory,Memory_wise_v3
from clustercontrast.utils import to_torch
warnings.filterwarnings("ignore")

import setproctitle

# 设置进程名
setproctitle.setproctitle("m2_v1")

best_mAP = 0
best_rank1 = 0
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_test_loader(dataset, height, width, batch_size, workers, testset=None, test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        VideoPreprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def calc_eps(rerank_dist):
    tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
    tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
    tri_mat = np.sort(tri_mat,axis=None)
    rho = 1.6e-3
    top_num = np.round(rho*tri_mat.size).astype(int)
    eps = tri_mat[:top_num].mean()
    # print('eps for cluster: {:.3f}'.format(eps))

    return eps

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                        num_instances, iters, trainset=None, no_cam=False, train_transformer=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(VideoPreprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                           num_instances, iters, trainset=None, no_cam=False, train_transformer=None,
                           train_transformer1=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(VideoPreprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(VideoPreprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,
                                          transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers


def calculate_cluster_accuracy(pseudo_labels, real_labels):
    """计算聚类准确率
    
    参数:
        pseudo_labels: 伪标签列表
        real_labels: 真实标签列表
        
    返回:
        accuracy: 聚类准确率
    """
    if len(pseudo_labels) == 0:
        return 0.0
    
    # 创建伪标签到真实标签的映射
    cluster_to_class = {}
    for pseudo_label, real_label in zip(pseudo_labels, real_labels):
        if pseudo_label == -1:
            continue
        if pseudo_label not in cluster_to_class:
            cluster_to_class[pseudo_label] = {}
        if real_label not in cluster_to_class[pseudo_label]:
            cluster_to_class[pseudo_label][real_label] = 0
        cluster_to_class[pseudo_label][real_label] += 1
    
    # 为每个聚类找到最常见的真实标签
    cluster_assignments = {}
    for cluster_id, class_counts in cluster_to_class.items():
        cluster_assignments[cluster_id] = max(class_counts.items(), key=lambda x: x[1])[0]
    
    # 计算正确分配的样本数
    correct_count = 0
    total_count = 0
    for pseudo_label, real_label in zip(pseudo_labels, real_labels):
        if pseudo_label == -1:
            continue
        total_count += 1
        if cluster_assignments.get(pseudo_label) == real_label:
            correct_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0

def main(config):
    log_s1_name = 'vcm2bupt_v1'
    main_worker_stage1(config, log_s1_name)

def main_worker_stage1(config, log_s1_name):
    start_epoch = 0
    best_rank1 = 0
    start_time = time.monotonic()

    #设置好日志
    output_path = os.path.join(config.output_path, log_s1_name)
    make_dirs(output_path)
    make_dirs(os.path.join(output_path, 'logs/'))
    make_dirs(os.path.join(output_path, 'models/'))
    # make_dirs(os.path.join(os.path.join(output_path, 'logs/'), 'log.txt'))
    
    logger = Logger(os.path.join(os.path.join(output_path, 'logs/'), 'log.txt'))
    logger('\n' * 3)
    logger(config)
    

    #加载数据集和模型
    iters = config.steps if (config.steps > 0) else None
    logger("==> Load unlabeled dataset")
    dataset_bupt = get_data(config.dataset,config.bupt_data_path)
    # dataset_vcm = get_data('vcm',config.vcm_data_path)
    model = Base(config,log_s1_name)
    loaders = Loader(config)
    model.resume_model('vcm/debug/UCP_V1/models')
    # #设置优化器
    scaler = amp.GradScaler()
    # params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1)

    # Trainer
    # trainer = ClusterContrastTrainer_ADCA_joint(model)

    # ########################
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    height = config.img_h
    width = config.img_w
    train_transformer_rgb = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability=0.5)
    ])

    train_transformer_rgb1 = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability=0.5),
        ChannelExchange(gray=2)  # 2
    ])

    transform_thermal = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((288, 144)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability=0.5),
        ChannelAdapGray(probability=0.5)])
    for epoch in range(config.total_train_epoch):
        dynamic_eps = epoch >= 15  # 控制在20轮之后自动计算eps
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = config.eps
                logger('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = config.eps
                logger('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            logger('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_bupt, config.img_h, config.img_w,
                                                 128, config.workers,
                                                 testset=sorted(dataset_bupt.train_rgb))
            features_rgb, features_rgb_frame,features_rgb_frame_together,features_rgb_together,labels_rgb = extract_features(model, cluster_loader_rgb, print_freq=50)
            features_rgb = torch.cat([features_rgb[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_rgb)], 0)
            features_rgb_frame = torch.cat([features_rgb_frame[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_rgb)], 0)
            features_rgb_frame_together = torch.cat([features_rgb_frame_together[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_rgb)], 0)
            features_rgb_together = torch.cat([features_rgb_together[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_rgb)], 0)
            labels_rgb = torch.cat([labels_rgb[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_rgb)], 0)
            del cluster_loader_rgb,
            
            # torch.save(features_rgb, 'features_rgb.pth')
            # torch.save(features_rgb_frame, 'features_rgb_frame.pth')

            # features_rgb_frame = torch.load('features_rgb_frame.pth')
            # features_rgb = torch.load('features_rgb.pth')


            rerank_dist_rgb = compute_jaccard_distance(features_rgb_together, k1=config.k1, k2=config.k2,
                                                       search_option=3)  # rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#

            
            # if dynamic_eps:
            #     eps_rgb = calc_eps(rerank_dist=rerank_dist_rgb)
            #     cluster_rgb = DBSCAN(eps=eps_rgb, min_samples=4, metric='precomputed', n_jobs=-1)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)


            # 提取features_ir帧级特征
            logger('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_bupt, config.img_h, config.img_w,
                                                128, config.workers,
                                                testset=sorted(dataset_bupt.train_ir))
            features_ir, features_ir_frame,features_ir_frame_together,features_ir_together,labels_ir = extract_features(model, cluster_loader_ir, print_freq=50)
            features_ir = torch.cat([features_ir[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_ir)], 0)
            features_ir_frame = torch.cat([features_ir_frame[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_ir)], 0)
            features_ir_frame_together = torch.cat([features_ir_frame_together[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_ir)], 0)
            features_ir_together = torch.cat([features_ir_together[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_ir)], 0)
            labels_ir = torch.cat([labels_ir[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_ir)], 0)
            del cluster_loader_ir,
            

            # torch.save(features_ir, 'features_ir.pth')
            # torch.save(features_ir_frame, 'features_ir_frame.pth')

            # features_ir_frame = torch.load('features_ir_frame.pth')
            # features_ir = torch.load('features_ir.pth')
            
            rerank_dist_ir = compute_jaccard_distance(features_ir_together, k1=30, k2=config.k2,
                                                      search_option=3)  # rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#

            
            # if dynamic_eps:
            #     eps_ir = calc_eps(rerank_dist=rerank_dist_ir)
            #     cluster_ir = DBSCAN(eps=eps_ir, min_samples=4, metric='precomputed', n_jobs=-1)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)

            
            del rerank_dist_rgb
            del rerank_dist_ir
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        cluster_features_ir_frame = generate_cluster_features(pseudo_labels_ir, features_ir_frame_together)
        cluster_features_rgb_frame = generate_cluster_features(pseudo_labels_rgb, features_rgb_frame_together)


        memory_ir = ClusterMemory(768, num_cluster_ir, temp=config.temp,
                                  momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_rgb = ClusterMemory(768, num_cluster_rgb, temp=config.temp,
                                   momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_ir_frame = ClusterMemory(768, num_cluster_ir, temp=config.temp,
                                  momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_rgb_frame = ClusterMemory(768, num_cluster_rgb, temp=config.temp,
                                   momentum=config.momentum, use_hard=config.use_hard).cuda()
        
        memory_rgb_instance =Memory_wise_v3(768, len(dataset_bupt.train_rgb),num_cluster_rgb,temp=config.temp, 
                            momentum=config.momentum).cuda()
        memory_ir_instance = Memory_wise_v3(768, len(dataset_bupt.train_ir), num_cluster_ir, temp=config.temp,
                                   momentum=config.momentum).cuda()        
        memory_rgb_frame_instance = Memory_wise_v3(768, len(dataset_bupt.train_rgb), num_cluster_rgb, temp=config.temp,
                                   momentum=config.momentum).cuda()
        memory_ir_frame_instance = Memory_wise_v3(768, len(dataset_bupt.train_ir), num_cluster_ir, temp=config.temp,
                                   momentum=config.momentum).cuda()
        

        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        memory_ir_frame.features = F.normalize(cluster_features_ir_frame, dim=1).cuda()
        memory_rgb_frame.features = F.normalize(cluster_features_rgb_frame, dim=1).cuda()
        memory_rgb_instance.features = F.normalize(features_rgb, dim=1).cuda()
        memory_ir_instance.features = F.normalize(features_ir, dim=1).cuda()        
        memory_rgb_frame_instance.features = F.normalize(features_rgb_frame_together, dim=1).cuda()
        memory_ir_frame_instance.features = F.normalize(features_ir_frame_together, dim=1).cuda()

        model.memory_ir = memory_ir
        model.memory_rgb = memory_rgb
        model.memory_ir_frame = memory_ir_frame
        model.memory_rgb_frame = memory_rgb_frame
        model.memory_rgb_instance = memory_rgb_instance
        model.memory_ir_instance = memory_ir_instance        
        model.memory_rgb_frame_instance = memory_rgb_frame_instance
        model.memory_ir_frame_instance = memory_ir_frame_instance

        model.memory_ir_instance.labels = torch.from_numpy(pseudo_labels_ir)
        model.memory_rgb_instance.labels = torch.from_numpy(pseudo_labels_rgb)

        pseudo_labeled_dataset_ir = []
        ir_label = []
        ir_real_labels = []  # 用于存储真实标签
        for i, ((fname, pid, cid), label) in enumerate(zip(sorted(dataset_bupt.train_ir), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
                ir_real_labels.append(pid)  # 存储真实标签
        logger('==> Statistics for IR epoch {}: {} clusters, Clustering Accuracy: {:.4f}'.format(
            epoch, num_cluster_ir, calculate_cluster_accuracy(ir_label, ir_real_labels)))
        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        rgb_real_labels = []  # 用于存储真实标签
        for i, ((fname, pid, cid), label) in enumerate(zip(sorted(dataset_bupt.train_rgb), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
                rgb_real_labels.append(pid)  # 存储真实标签
        logger('==> Statistics for RGB epoch {}: {} clusters, Clustering Accuracy: {:.4f}'.format(
            epoch, num_cluster_rgb, calculate_cluster_accuracy(rgb_label, rgb_real_labels)))


        # if epoch >= config.ht:
        #     with torch.no_grad():
        #         # 跨模态交互步骤
        #         TOPK = 20
        #         Score_TOPK = 20
                
        #         # 计算RGB特征和IR特征之间的相似度
        #         # 序列级特征相似度
        #         seq_sim_rgb_ir = F.normalize(features_rgb, dim=1).mm(F.normalize(features_ir, dim=1).t())
        #         # 帧级特征相似度
        #         frame_sim_rgb_ir = F.normalize(features_rgb_frame_together, dim=1).mm(F.normalize(features_ir_frame_together, dim=1).t())
                
        #         # 为每个RGB样本找到最相似的IR样本
        #         _, seq_indices_rgb_ir = torch.topk(seq_sim_rgb_ir, int(Score_TOPK))
        #         _, frame_indices_rgb_ir = torch.topk(frame_sim_rgb_ir, int(Score_TOPK))
                
        #         # 获取IR样本的标签
        #         ir_labels_tensor = model.memory_ir_instance.labels.detach().cpu()
        #         seq_label_candidates = ir_labels_tensor[seq_indices_rgb_ir]
        #         frame_label_candidates = ir_labels_tensor[frame_indices_rgb_ir]
                
        #         # 统计标签匹配程度
        #         intersect_count_list = []
        #         for k in range(TOPK):
        #             # 找到序列级和帧级都认为匹配的标签
        #             intersect_count = (frame_label_candidates == seq_label_candidates[:,k].view(-1,1)).int().sum(1).view(-1,1)
        #             intersect_count_list.append(intersect_count)
                
        #         # 合并统计结果并找到最佳匹配
        #         intersect_count_tensor = torch.cat(intersect_count_list, 1)
        #         _, best_match_indices = torch.topk(intersect_count_tensor, 1)
        #         best_ir_labels = torch.gather(seq_label_candidates, dim=1, index=best_match_indices)
                
        #         # 结构平滑 - 利用RGB样本的邻域关系优化标签分配
        #         rgb_cm_label = best_ir_labels.view(-1) + 1  # 加1避免-1标签
        #         rgb_cm_label_onehot = F.one_hot(rgb_cm_label.long(), int(num_cluster_ir)+1).float()
                
        #         # 计算RGB样本之间的相似度
        #         rgb_self_sim_seq = torch.mm(F.normalize(features_rgb, dim=1), F.normalize(features_rgb, dim=1).t())
        #         rgb_self_sim_frame = torch.mm(F.normalize(features_rgb_frame_together, dim=1), F.normalize(features_rgb_frame_together, dim=1).t())
        #         rgb_self_sim = rgb_self_sim_seq + rgb_self_sim_frame
                
        #         # 为每个样本保留最相似的K个样本
        #         k_neighbors = 5
        #         topk_sim, indices_topk = torch.topk(rgb_self_sim, k_neighbors)
        #         neighbor_mask = torch.zeros_like(rgb_self_sim)
        #         neighbor_mask = neighbor_mask.scatter(1, indices_topk, 1)
                
        #         # 基于邻域结构平滑标签
        #         smooth_rgb_labels = torch.mm(neighbor_mask, rgb_cm_label_onehot)
        #         smooth_rgb_labels = torch.argmax(smooth_rgb_labels, dim=1).view(-1).cpu().numpy()
                
        #         # 转回原始标签格式（减1）
        #         pseudo_labels_rgb_cm = np.array([int(label-1) for label in smooth_rgb_labels])
                
        #         # 更新RGB伪标签
        #         pseudo_labels_rgb = pseudo_labels_rgb_cm
                
        #         # 计算更新后的聚类数量
        #         num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
        #         logger('==> Updated Statistics for RGB epoch {}: {} clusters after cross-modal interaction'.format(epoch, num_cluster_rgb))
                
        #         # 重建数据集 - 这是关键步骤，确保标签为-1的样本被筛选掉
        #         pseudo_labeled_dataset_ir = []
        #         for i, ((fname, pid, cid), label) in enumerate(zip(sorted(dataset_bupt.train_ir), pseudo_labels_ir)):
        #             # 获取原始标签
        #             orig_label = model.memory_ir_instance.labels[i].item()
        #             # 同时过滤当前伪标签为-1和原始标签为-1的样本
        #             if (label != -1) and (orig_label != -1):
        #                 pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                
        #         pseudo_labeled_dataset_rgb = []
        #         for i, ((fname, pid, cid), label) in enumerate(zip(sorted(dataset_bupt.train_rgb), pseudo_labels_rgb)):
        #             # 获取原始标签
        #             orig_label = model.memory_rgb_instance.labels[i].item()
        #             # 同时过滤当前伪标签为-1和原始标签为-1的样本
        #             if (label != -1) and (orig_label != -1):
        #                 pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                
        #         # 合并特征和标签
        #         features_combined = torch.cat((features_rgb, features_ir), dim=0)
        #         pseudo_labels_combined = np.concatenate((pseudo_labels_rgb, pseudo_labels_ir))
                
        #         # 生成聚类中心特征
        #         cluster_features_combined = generate_cluster_features(pseudo_labels_combined, features_combined)
                
        #         # 创建共享内存
        #         shared_memory = ClusterMemory(768, num_cluster_ir, temp=config.temp, 
        #                                      momentum=0.1, use_hard=config.use_hard).cuda()
        #         shared_memory.features = F.normalize(cluster_features_combined, dim=1).cuda()
                
        #         # 帧级特征也做类似处理
        #         features_frame_combined = torch.cat((features_rgb_frame_together, features_ir_frame_together), dim=0)
        #         cluster_features_frame_combined = generate_cluster_features(pseudo_labels_combined, features_frame_combined)
        #         shared_memory_frame = ClusterMemory(768, num_cluster_ir, temp=config.temp, 
        #                                           momentum=0.1, use_hard=config.use_hard).cuda()
        #         shared_memory_frame.features = F.normalize(cluster_features_frame_combined, dim=1).cuda()
                
        #         # 更新模型的内存
        #         model.memory_ir = shared_memory
        #         model.memory_rgb = shared_memory
        #         model.memory_ir_frame = shared_memory_frame
        #         model.memory_rgb_frame = shared_memory_frame
        
        train_loader_ir = get_train_loader_ir(config, dataset_bupt, config.img_h, config.img_w,
                                              4, config.workers, config.num_instances, iters,
                                              trainset=pseudo_labeled_dataset_ir, no_cam=config.no_cam,
                                              train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(config, dataset_bupt, config.img_h, config.img_w,
                                                  2, config.workers, config.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_rgb, no_cam=config.no_cam,
                                                  train_transformer=train_transformer_rgb,
                                                  train_transformer1=train_transformer_rgb1)
        
        model.train_loader_ir = train_loader_ir
        model.train_loader_rgb = train_loader_rgb

        _, result = train(model, loaders, config,scaler,epoch)
        logger('Time: {}; Epoch: {}; {}; {}'.format(time_now(), epoch, result,model.model_optimizer.state_dict()['param_groups'][0]['lr']))




        # Eval/Test
        if epoch >= 0 and ((epoch + 1) % config.eval_epoch == 0 or (epoch == config.total_train_epoch - 1)):
            if config.dataset=='bupt':
                cmc, mAP,eval_str = test_bupt(model, loaders, config)
                is_best_rank = (cmc[1][0] >= best_rank1)
                best_rank1 = max(cmc[1][0], best_rank1)
                model.save_model(epoch, is_best_rank)
                logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str[0],eval_str[1]))
            elif config.dataset=='vcm':
                cmc, mAP,cmc2, mAP2, eval_str_t2v,eval_str_v2t = test_vcm(model, loaders, t2v=True)
                # _, _, eval_str_v2t = test_vcm(model, loaders, t2v=False)
                is_best_rank = (cmc[0] >= best_rank1)
                best_rank1 = max(cmc[0], best_rank1)
                model.save_model(epoch, is_best_rank)
                logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str_t2v,eval_str_v2t))

    end_time = time.monotonic()
    logger('Total running time:{} '.format(timedelta(seconds=end_time-start_time)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--module', type=str, default='video_B_tri', help='B')
    parser.add_argument('--dataset', default='bupt', help='dataset name: vcm , bupt]')
    parser.add_argument('--vcm_data_path', type=str, default='/data1/wangzj/datasets/')
    parser.add_argument('--bupt_data_path', type=str, default='/data1/wangzj/datasets/')
    parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')
    parser.add_argument('--num_pos', default=4, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--cls_cue',type=ast.literal_eval,default=True)
    parser.add_argument('--prompt_cue',type=ast.literal_eval,default=True)
    parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pid_num', type=int, default=500) #bupt 1074;vcm 500
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=0.00025)
    parser.add_argument('--lr_times', type=int, default=25)
    parser.add_argument('--num_workers', default=8, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--lower', type=float, default=0.02)
    parser.add_argument('--upper', type=float, default=0.4)
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--output_path', type=str, default='bupt/debug/vcm2bupt_m2',
                        help='path to save related informations')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=False)
    parser.add_argument('--total_train_epoch', type=int, default=50)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--vcm_test_mode', default='t2v', help='dataset name: regdb or sysu,vcm]')
    parser.add_argument('--resume_test_model', type=int, default=119, help='-1 for no resuming')
    parser.add_argument('--STH_start_layer', type=int, default=9, help='-1 for no resuming')
    parser.add_argument('--prompt_index', type=int, default=3, help='-1 for no resuming')


    ###bupt
    parser.add_argument('--train_frame_sample', type=str, default='random')
    parser.add_argument('--sequence_length', type=int, default=6)
    parser.add_argument('--random_flip', action='store_false', default=True)
    parser.add_argument('--fake', action='store_true', default=False)
    parser.add_argument('--test_frame_sample', type=str, default='uniform')
    parser.add_argument('--test_sampler', type=str,
                             default='ConsistentModalitySampler', help='None for no shuffle')
    parser.add_argument('--test_bs', type=int, default=64)  # Please don't change it.
    parser.add_argument('--train_sampler', type=str,
                             default='RandomIdentitySampler', help='None for shuffle')
    parser.add_argument('--train_bs', type=int, default=16)
    parser.add_argument('--train_sampler_nc', type=int, default=2)
    parser.add_argument('--train_sampler_nt', type=int, default=1)
    parser.add_argument('--max_rank', type=int, default=20)

    #loss
    parser.add_argument('--weight_a', type=float, default=0.08)
    parser.add_argument('--weight_b', type=float, default=0.4)
    parser.add_argument('--weight_c', type=float, default=1.0)

    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")

    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--seq_lenth', type=int, default=6)
    parser.add_argument('--num-instances', type=int, default=4,
                    help="each minibatch consist of "
                            "(batch_size // num_instances) identities, and "
                            "each identity has num_instances instances, "
                            "default: 0 (NOT USE)")
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--no-cam', action="store_true")
    #邻域学习
    parser.add_argument('--hm', type=int, default=0)#同模态
    parser.add_argument('--ht', type=int, default=10)#跨模态
    parser.add_argument('--hm_thresh', type=float, default=0.9)
    parser.add_argument('--ht_thresh', type=float, default=0.9) #阈值



    config = parser.parse_args()
    seed_torch(config.seed)
    main(config)
