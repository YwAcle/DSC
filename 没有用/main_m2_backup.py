import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
from clustercontrast.models.cm import ClusterMemory,EnhancedClusterMemory
warnings.filterwarnings("ignore")

import setproctitle
from sklearn.metrics.pairwise import cosine_similarity

# 设置进程名
setproctitle.setproctitle("m2")

best_mAP = 0
best_rank1 = 0
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
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

def select_merge_data(dists):
    dists = torch.from_numpy(dists)
    print(dists.size())
    dists = dists.numpy()
    ind = np.unravel_index(np.argsort(dists, axis=None)[::-1], dists.shape)  # np.argsort(dists, axis=1)#
    idx1 = ind[0]
    idx2 = ind[1]
    dist_list = dists[idx1, idx2]
    return idx1, idx2, dist_list

def compute_combined_distance(features_distribution, features, epoch, k1=30, k2=6, search_option=3):
    """
    融合特征距离和概率分布距离的联合度量
    
    Args:
        features_distribution: 分类器输出的概率分布
        features: 原始特征
        epoch: 当前训练轮次，用于动态调整权重
        k1, k2: Jaccard距离计算参数
    """
    # 计算特征的Jaccard距离
    feature_dist = compute_jaccard_distance(features, k1=k1, k2=k2, search_option=search_option)
    
    # 计算概率分布的Jaccard距离
    prob_dist = compute_jaccard_distance(features_distribution, k1=k1, k2=k2, search_option=search_option)
    
    # 动态权重计算: 从0.8线性衰减到0.3
    total_epochs = 30  # 总训练轮数
    min_alpha = 0.3
    max_alpha = 0.8
    alpha = max(min_alpha, max_alpha - (max_alpha - min_alpha) * epoch / total_epochs)
    
    # 融合两种距离
    combined_dist = alpha * feature_dist + (1 - alpha) * prob_dist
    
    return combined_dist


def main(config):
    log_s1_name = 'vcm2bupt_s1'
    log_s2_name = 'vcm2bupt_s2'
    main_worker_stage1(config, log_s1_name)
    main_worker_stage2(config, log_s1_name, log_s2_name) 

def main_worker_stage1(config, log_s1_name):
    start_epoch = 0
    best_rank1 = 0
    start_time = time.monotonic()
    cudnn.benchmark = True

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
    iters = config.iters if (config.iters > 0) else None
    logger("==> Load unlabeled dataset")
    dataset_bupt = get_data(config.dataset,config.bupt_data_path)
    # dataset_vcm = get_data('vcm',config.vcm_data_path)
    Model = Base(config,log_s1_name)
    loaders = Loader(config)
    Model.resume_model('vcm/debug/UCP_V1/models')
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
        # dynamic_eps = epoch >= 20  # 控制在20轮之后自动计算eps
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
                                                 256, config.workers,
                                                 testset=sorted(dataset_bupt.train_rgb))
            
            features_rgb,features_rgb_frame = extract_features(Model, cluster_loader_rgb, print_freq=50)
            features_rgb = torch.cat([features_rgb[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_rgb)], 0)
            features_rgb_frame = torch.cat([features_rgb_frame[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_rgb)], 0)
            del cluster_loader_rgb,
            
            # features_rgb = torch.load('features_rgb.pth')
            # torch.save(features_rgb, 'features_rgb.pth')
            # 处理RGB帧级特征
            num_sequence, seq_length, dim = features_rgb_frame.shape
            features_rgb_frame_ = features_rgb_frame.reshape(num_sequence * seq_length, dim)

            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=config.k1, k2=config.k2,
                                                       search_option=3)  # rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#

            rerank_dist_rgb_frame = compute_jaccard_distance(features_rgb_frame_, k1=config.k1, k2=config.k2,
                                                       search_option=3)  # rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            # if dynamic_eps:
            #     eps_rgb = calc_eps(rerank_dist=rerank_dist_rgb)
            #     cluster_rgb = DBSCAN(eps=eps_rgb, min_samples=4, metric='precomputed', n_jobs=-1)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            pseudo_labels_rgb_frame = cluster_rgb.fit_predict(rerank_dist_rgb_frame)



            # 提取features_ir帧级特征
            logger('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_bupt, config.img_h, config.img_w,
                                                256, config.workers,
                                                testset=sorted(dataset_bupt.train_ir))
            features_ir, features_ir_frame = extract_features(Model, cluster_loader_ir, print_freq=50)
            features_ir = torch.cat([features_ir[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_ir)], 0)
            features_ir_frame = torch.cat([features_ir_frame[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_ir)], 0)
            
            # 将帧级特征从(num_sequence,6,dim)展开为(num_sequence*6,dim)
            # 原始形状: (num_sequence, 6, dim)
            num_sequence, seq_length, dim = features_ir_frame.shape
            # 展开后形状: (num_sequence*6, dim)
            features_ir_frame_ = features_ir_frame.reshape(num_sequence * seq_length, dim)
            
            del cluster_loader_ir,
            
            # torch.save(features_ir, 'features_ir.pth')
            # features_ir = torch.load('features_ir.pth')
            
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=config.k1, k2=config.k2,
                                                       search_option=3)  # rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            rerank_dist_ir_frame = compute_jaccard_distance(features_ir_frame_, k1=config.k1, k2=config.k2,
                                                       search_option=3)  # rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            # if dynamic_eps:
            #     eps_ir = calc_eps(rerank_dist=rerank_dist_ir)
            #     cluster_ir = DBSCAN(eps=eps_ir, min_samples=4, metric='precomputed', n_jobs=-1)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            pseudo_labels_ir_frame = cluster_ir.fit_predict(rerank_dist_ir_frame)
            
            # 帧级伪标签自纠正 - IR
            logger('==> Self-correction for IR frame-level pseudo labels')
            seq_length = 6  # 每个序列的帧数
            num_sequences = len(pseudo_labels_ir_frame) // seq_length
            corrected_pseudo_labels_ir_frame = pseudo_labels_ir_frame.copy()
            
            for i in range(num_sequences):
                start_idx = i * seq_length
                end_idx = start_idx + seq_length
                seq_labels = pseudo_labels_ir_frame[start_idx:end_idx]
                
                # 统计序列中各标签出现次数
                label_counts = Counter(seq_labels)
                # 移除-1标签(噪声)，避免其影响多数投票
                if -1 in label_counts and len(label_counts) > 1:
                    del label_counts[-1]
                
                # 如果序列中有标签，找出出现最多的标签
                if label_counts:
                    majority_label = label_counts.most_common(1)[0][0]
                    # 直接使用出现次数最多的标签进行修正，不需要满足超过一半的条件
                    corrected_pseudo_labels_ir_frame[start_idx:end_idx] = majority_label
            
            # 使用修正后的帧级伪标签
            pseudo_labels_ir_frame = corrected_pseudo_labels_ir_frame
            
            # 对RGB数据执行相同的自纠正过程
            logger('==> Self-correction for RGB frame-level pseudo labels')
            corrected_pseudo_labels_rgb_frame = pseudo_labels_rgb_frame.copy()
            
            for i in range(num_sequences):
                start_idx = i * seq_length
                end_idx = start_idx + seq_length
                seq_labels = pseudo_labels_rgb_frame[start_idx:end_idx]
                
                # 统计序列中各标签出现次数
                label_counts = Counter(seq_labels)
                # 移除-1标签(噪声)，避免其影响多数投票
                if -1 in label_counts and len(label_counts) > 1:
                    del label_counts[-1]
                
                # 如果序列中有标签，找出出现最多的标签
                if label_counts:
                    majority_label = label_counts.most_common(1)[0][0]
                    # 直接使用出现次数最多的标签进行修正，不需要满足超过一半的条件
                    corrected_pseudo_labels_rgb_frame[start_idx:end_idx] = majority_label
            
            # 使用修正后的帧级伪标签
            pseudo_labels_rgb_frame = corrected_pseudo_labels_rgb_frame
            
            
            del rerank_dist_rgb
            del rerank_dist_ir
            del rerank_dist_rgb_frame
            del rerank_dist_ir_frame
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
            num_cluster_ir_frame = len(set(pseudo_labels_ir_frame)) - (1 if -1 in pseudo_labels_ir_frame else 0)
            num_cluster_rgb_frame = len(set(pseudo_labels_rgb_frame)) - (1 if -1 in pseudo_labels_rgb_frame else 0)
            

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
        cluster_features_rgb_frame = generate_cluster_features(pseudo_labels_rgb_frame,features_rgb_frame_)
        cluster_features_ir_frame = generate_cluster_features(pseudo_labels_ir_frame,features_ir_frame_)
        memory_ir = EnhancedClusterMemory(1536, num_cluster_ir, classifier=Model.model.classifier_ir,classifier_cue=Model.model.classifier_ir_cue, temp=config.temp,
                                  momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_rgb = EnhancedClusterMemory(1536, num_cluster_rgb, classifier=Model.model.classifier_rgb,classifier_cue=Model.model.classifier_rgb_cue, temp=config.temp,
                                   momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_rgb_frame = ClusterMemory(1536,num_cluster_rgb_frame,temp=config.temp,
                                  momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_ir_frame = ClusterMemory(1536,num_cluster_ir_frame,temp=config.temp,
                                  momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        memory_ir_frame.features = F.normalize(cluster_features_ir_frame,dim = 1).cuda()
        memory_rgb_frame.features = F.normalize(cluster_features_rgb_frame,dim = 1).cuda()

        # trainer.memory_ir = memory_ir
        # trainer.memory_rgb = memory_rgb

        # 原有的视频级伪标签数据集生成
        pseudo_labeled_dataset_ir = []
        ir_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_bupt.train_ir), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        logger('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))
        
        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_bupt.train_rgb), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
        logger('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))
        
        # 为帧级特征生成伪标签数据集和训练加载器
        logger('==> Generating frame-level pseudo-labeled datasets')
        pseudo_labeled_dataset_ir_frame = []
        ir_frame_label = []
        seq_length = 6  # 每个序列的帧数
        
        # 由于自纠正后每6个帧的伪标签都是一致的，只需要每隔6个取一个伪标签
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_bupt.train_ir), pseudo_labels_ir_frame[::seq_length])):
            if label != -1:
                # 直接使用原始文件名，不需要添加帧索引
                pseudo_labeled_dataset_ir_frame.append((fname, label.item(), cid))
                ir_frame_label.append(label.item())
        
        pseudo_labeled_dataset_rgb_frame = []
        rgb_frame_label = []
        # 同样，对RGB数据也是每隔6个取一个伪标签
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_bupt.train_rgb), pseudo_labels_rgb_frame[::seq_length])):
            if label != -1:
                # 直接使用原始文件名，不需要添加帧索引
                pseudo_labeled_dataset_rgb_frame.append((fname, label.item(), cid))
                rgb_frame_label.append(label.item())
        
        num_cluster_ir_frame = len(set(ir_frame_label))
        num_cluster_rgb_frame = len(set(rgb_frame_label))
        logger('==> Statistics for IR frame-level epoch {}: {} clusters'.format(epoch, num_cluster_ir_frame))
        logger('==> Statistics for RGB frame-level epoch {}: {} clusters'.format(epoch, num_cluster_rgb_frame))
        
        # 创建原有的视频级训练加载器
        train_loader_ir = get_train_loader_ir(config, dataset_bupt, config.img_h, config.img_w,
                                              4, config.workers, config.num_instances, iters,
                                              trainset=pseudo_labeled_dataset_ir, no_cam=config.no_cam,
                                              train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(config, dataset_bupt, config.img_h, config.img_w,
                                                  2, config.workers, config.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_rgb, no_cam=config.no_cam,
                                                  train_transformer=train_transformer_rgb,
                                                  train_transformer1=train_transformer_rgb1)
        
        # 创建帧级训练加载器
        # 注意：这里需要修改或创建新的数据加载函数来处理帧级数据
        train_loader_ir_frame = get_train_loader_ir(config, dataset_bupt, config.img_h, config.img_w,
                                                   4, config.workers, config.num_instances, iters,
                                                   trainset=pseudo_labeled_dataset_ir_frame, no_cam=config.no_cam,
                                                   train_transformer=transform_thermal)

        train_loader_rgb_frame = get_train_loader_color(config, dataset_bupt, config.img_h, config.img_w,
                                                       2, config.workers, config.num_instances, iters,
                                                       trainset=pseudo_labeled_dataset_rgb_frame, no_cam=config.no_cam,
                                                       train_transformer=train_transformer_rgb,
                                                       train_transformer1=train_transformer_rgb1)

        _, result = train(Model, loaders, config,scaler,memory_ir,memory_rgb,memory_ir_frame,memory_rgb_frame,train_loader_ir=train_loader_ir,train_loader_rgb=train_loader_rgb)
        logger('Time: {}; Epoch: {}; {}; {}'.format(time_now(), epoch, result,Model.model_optimizer.state_dict()['param_groups'][0]['lr']))



        # train_loader_ir.new_epoch()
        # train_loader_rgb.new_epoch()

        # trainer.train(epoch, train_loader_ir, train_loader_rgb, optimizer, print_freq=config.print_freq,
        #               train_iters=len(train_loader_ir))

        # Eval/Test
        if epoch >= 0 and ((epoch + 1) % config.eval_step == 0 or (epoch == config.total_train_epoch - 1)):
            if config.dataset=='bupt':
                cmc, mAP,eval_str = test_bupt(Model, loaders, config)
                is_best_rank = (cmc[1][0] >= best_rank1)
                best_rank1 = max(cmc[1][0], best_rank1)
                Model.save_model(epoch, is_best_rank)
                logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str[0],eval_str[1]))
            elif config.dataset=='vcm':
                cmc, mAP,cmc2, mAP2, eval_str_t2v,eval_str_v2t = test_vcm(Model, loaders, t2v=True)
                # _, _, eval_str_v2t = test_vcm(model, loaders, t2v=False)
                is_best_rank = (cmc[0] >= best_rank1)
                best_rank1 = max(cmc[0], best_rank1)
                Model.save_model(epoch, is_best_rank)
                logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str_t2v,eval_str_v2t))

    end_time = time.monotonic()
    logger('Total running time:{} '.format(timedelta(seconds=end_time-start_time)))

def main_worker_stage2(config, log_s1_name, log_s2_name):
    start_epoch = 0
    best_rank1 = 0
    start_time = time.monotonic()
    cudnn.benchmark = True

    #设置好日志
    output_path = os.path.join(config.output_path, log_s2_name)
    make_dirs(output_path)
    make_dirs(os.path.join(output_path, 'logs/'))
    make_dirs(os.path.join(output_path, 'models/'))
    logger = Logger(os.path.join(os.path.join(output_path, 'logs/'), 'log.txt'))
    logger('\n' * 3)
    logger(config)
    

    #加载数据集和模型
    iters = config.iters if (config.iters > 0) else None
    logger("==> Load unlabeled dataset")
    dataset_bupt = get_data(config.dataset,config.bupt_data_path)
    dataset_vcm = get_data('vcm',config.vcm_data_path)
    Model = Base(config,log_s2_name)
    loaders = Loader(config)
    Model.resume_model('bupt/debug/vcm2bupt_m2_backup/vcm2bupt_s1/models')
    scaler = amp.GradScaler()


    for epoch in range(config.total_train_epoch):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = config.eps
                logger('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = config.eps  # +0.1
                logger('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            logger('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_bupt, config.img_h, config.img_w,
                                                 32, config.workers,
                                                 testset=sorted(dataset_bupt.train_rgb))
            features_rgb, _ = extract_features(Model, cluster_loader_rgb, print_freq=50)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_rgb)], 0)
            logger('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_bupt, config.img_h, config.img_w,
                                                32, config.workers,
                                                testset=sorted(dataset_bupt.train_ir))
            features_ir, _ = extract_features(Model, cluster_loader_ir, print_freq=50)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f[0]].unsqueeze(0) for f, _, _ in sorted(dataset_bupt.train_ir)], 0)

            # features_rgb = torch.load('features_rgb.pth')
            # features_ir = torch.load('features_ir.pth')

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=config.k1, k2=config.k2,
                                                      search_option=3)  # rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=config.k1, k2=config.k2,
                                                       search_option=3)  # rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
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
        # del features, cluster_loader_1,cluster_loader_2,cluster_loader_3,cluster_loader_4,features_1,features_2,features_3,features_4
        # del features_ir,features_rgb
        # Create hybrid memory
        memory_ir = ClusterMemory(768, num_cluster_ir, temp=config.temp,
                                  momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_rgb = ClusterMemory(768, num_cluster_rgb, temp=config.temp,
                                   momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        pseudo_labeled_dataset_ir = []
        ir_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_bupt.train_ir), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        logger('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_bupt.train_rgb), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
        logger('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        ###################################CMA
        logger('merge ir and rgb momery'.format(epoch, num_cluster_ir))
        logger('select_merge_data')
        label_to_images = {}
        dist_cm = np.matmul(features_rgb.numpy(), np.transpose(features_ir.numpy()))
        idx1, idx2, dist_list = select_merge_data(dist_cm)
        del features_ir, features_rgb

        del dist_cm
        rgb_label_cnt = Counter(rgb_label)
        ir_label_cnt = Counter(ir_label)
        idx_lenth = np.sum(dist_list >= 0.5)
        dist_list = dist_list[:idx_lenth]
        rgb2ir_label = [(i, j) for i, j in zip(np.array(pseudo_labels_rgb)[idx1[:idx_lenth]],
                                               np.array(pseudo_labels_ir)[idx2[:idx_lenth]])]
        # print('rgb2ir_label',rgb2ir_label)
        rgb2ir_label_cnt = Counter(rgb2ir_label)
        rgb2ir_label_cnt_sorted = sorted(rgb2ir_label_cnt.items(), key=lambda x: x[1], reverse=True)
        lenth = len(rgb2ir_label_cnt_sorted)
        lamda_cm = 0.1
        in_rgb_label = []
        in_ir_label = []
        match_cnt = 0
        right = 0
        lenth_ratio = 1.0
        for i in range(int(lenth * lenth_ratio)):
            key = rgb2ir_label_cnt_sorted[i][0]
            value = rgb2ir_label_cnt_sorted[i][1]
            if key[0] == -1 or key[1] == -1:
                continue
            if key[0] in in_rgb_label or key[1] in in_ir_label:
                continue
            update_memory = memory_ir.features[key[1]]
            memory_rgb.features[key[0]] = lamda_cm * memory_rgb.features[key[0]] + (1 - lamda_cm) * (
                update_memory)
            memory_ir.features[key[1]] = lamda_cm * memory_ir.features[key[1]] + (1 - lamda_cm) * (
                update_memory)
            in_rgb_label.append(key[0])
            in_ir_label.append(key[1])
        ########################
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
            ChannelExchange(gray=2)
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

        train_loader_ir = get_train_loader_ir(config, dataset_bupt, config.img_h, config.img_w,
                                              8, config.workers, config.num_instances, iters,
                                              trainset=pseudo_labeled_dataset_ir, no_cam=config.no_cam,
                                              train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(config, dataset_bupt, config.img_h, config.img_w,
                                                  4, config.workers, config.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_rgb, no_cam=config.no_cam,
                                                  train_transformer=train_transformer_rgb,
                                                  train_transformer1=train_transformer_rgb1)

        _, result = train(Model, loaders, config,scaler,memory_ir,memory_rgb,train_loader_ir=train_loader_ir,train_loader_rgb=train_loader_rgb)
        # Logger('Time: {}; Epoch: {}; {}; {}'.format(time_now(), epoch, result,model.model_optimizer.state_dict()['param_groups'][0]['lr']))
        
        logger('Time: {}; Epoch: {}; {}; {}'.format(
            time_now(), 
            epoch, 
            result, 
            Model.model_optimizer.state_dict()['param_groups'][0]['lr']
        ))



        # train_loader_ir.new_epoch()
        # train_loader_rgb.new_epoch()

        # trainer.train(epoch, train_loader_ir, train_loader_rgb, optimizer, print_freq=config.print_freq,
        #               train_iters=len(train_loader_ir))

        # Eval/Test
        if epoch >= 6 and ((epoch + 1) % config.eval_step == 0 or (epoch == config.total_train_epoch - 1)):
            if config.dataset=='bupt':
                cmc, mAP,eval_str = test_bupt(Model, loaders, config)
                is_best_rank = (cmc[1][0] >= best_rank1)
                best_rank1 = max(cmc[1][0], best_rank1)
                Model.save_model(epoch, is_best_rank)
                # Logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str[0],eval_str[1]))
                logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str[0],eval_str[1]))
            elif config.dataset=='vcm':
                cmc, mAP,cmc2, mAP2, eval_str_t2v,eval_str_v2t = test_vcm(Model, loaders, t2v=True)
                # _, _, eval_str_v2t = test_vcm(model, loaders, t2v=False)
                is_best_rank = (cmc[0] >= best_rank1)
                best_rank1 = max(cmc[0], best_rank1)
                Model.save_model(epoch, is_best_rank)
                # Logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str_t2v,eval_str_v2t))
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
    parser.add_argument('--pid_num', type=int, default=500) #bupt 1074
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=0.000025)
    parser.add_argument('--lr_times', type=int, default=25)
    parser.add_argument('--num_workers', default=8, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--lower', type=float, default=0.02)
    parser.add_argument('--upper', type=float, default=0.4)
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--output_path', type=str, default='bupt/debug/vcm2bupt_m2_backup',
                        help='path to save related informations')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=False)
    parser.add_argument('--total_train_epoch', type=int, default=30)
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
    parser.add_argument('--iters', type=int, default=200)

    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--seq_lenth', type=int, default=6)
    # parser.add_argument('--in_dim', type=int, defaul=2048)
    parser.add_argument('--c_learning_rate', type=float, default=0.0007)
    parser.add_argument('--num-instances', type=int, default=4,
                    help="each minibatch consist of "
                            "(batch_size // num_instances) identities, and "
                            "each identity has num_instances instances, "
                            "default: 0 (NOT USE)")
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--no-cam', action="store_true")
    parser.add_argument('--eval-step', type=int, default=2)



    config = parser.parse_args()
    seed_torch(config.seed)
    main(config)
