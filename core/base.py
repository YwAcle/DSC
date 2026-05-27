import os
import torch
import torch.nn as nn
import torch.optim as optim

from bisect import bisect_right
from network import  Classifier,Model
from tools import os_walk, TripletLoss_WRT,SupConLoss
from network.lr_scheduler import WarmupMultiStepLR
from network.scheduler import CosineLRScheduler
class Base:
    def __init__(self, config,log_name):
        self.config = config
        self.log_name = log_name
        self.pid_num = config.pid_num
        self.module = config.module

        self.max_save_model_num = config.max_save_model_num
        self.output_path = os.path.join(config.output_path, log_name)
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')

        self.learning_rate = config.learning_rate
        self.lr_times = config.lr_times
        self.weight_decay = config.weight_decay
        # self.milestones = config.milestones
        self.steps = config.steps

        self.img_h = config.img_h
        self.img_w = config.img_w
        self.seq_lenth = config.sequence_length
        self.weight_a = config.weight_a
        self.weight_b = config.weight_b
        self.weight_c = config.weight_c
        self.STH_start_layer = config.STH_start_layer

        self.train_loader_ir = None
        self.train_loader_rgb = None
        self.memory_ir = None
        self.memory_rgb = None
        self.memory_ir_frame = None
        self.memory_rgb_frame = None

        self.memory_rgb_instance = None
        self.memory_ir_instance = None
        self.memory_rgb_frame_instance = None
        self.memory_ir_frame_instance = None
        

        self.hm = config.hm
        self.ht = config.ht
        self.hm_thresh = config.hm_thresh
        self.ht_thresh = config.ht_thresh

        self._init_device()
        self._init_model()
        self._init_creiteron()
        self._init_optimizer()
        # self._init_optimzer_and_lr_scheduler()

    def _init_device(self):
        self.device = torch.device('cuda')


    # def _init_optimzer_and_lr_scheduler(self):
    #     optimizers = []
    #     lr_schedulers = []
    #     keys_dict = self.__dict__
    #     for key in keys_dict:
    #         if 'optimizer' in key:
    #             optimizers.append(keys_dict[key])
    #         if 'lr_scheduler' in key:
    #             lr_schedulers.append(keys_dict[key])
    #     self.optimizers = optimizers
    #     self.lr_schedulers = lr_schedulers

    def _init_model(self):

        self.model = Model(self.config,self.pid_num, self.img_h, self.img_w)
        self.model = self.model.to(self.device)


    def _init_creiteron(self):
        self.pid_creiteron = nn.CrossEntropyLoss()
        self.tri_creiteron = TripletLoss_WRT()
        self.con_creiteron = SupConLoss(self.device)




    # def _init_optimizer(self):
    #     params = []
    #     keys = []
    #     for key, value in self.model.named_parameters():
    #         lr = self.learning_rate
    #         if 'classifier' in key:
    #             lr = self.learning_rate * 2
    #         params += [{'params': [value], 'lr': lr, 'weight_decay': self.weight_decay}]
    #         keys += [[key]]
    #
    #     self.model_optimizer = getattr(torch.optim, 'Adam')(params)
    #     self.model_lr_scheduler = WarmupMultiStepLR(self.model_optimizer, self.milestones,
    #                                                        gamma=0.1, warmup_factor=0.01, warmup_iters=10)



    def _init_optimizer(self):
        params = []
        keys = []
        for key, value in self.model.named_parameters():
            if "text_encoder" in key:
                value.requires_grad_(False)
                continue
            if not value.requires_grad:
                continue
            lr = self.learning_rate
            weight_decay = self.weight_decay
            if "bias" in key:
                lr = self.learning_rate * self.lr_times
                weight_decay = self.weight_decay

            if "classifier" in key or "arcface" in key:
                lr = self.learning_rate * self.lr_times
                print('Using two times learning rate for fc ')

            if "prompt_learner" in key:
                lr = self.learning_rate * self.lr_times
                print('Using two times learning rate for fc ')

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

        self.model_optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_min = self.learning_rate*0.01
        warmup_lr_init = self.learning_rate*0.01
        warmup_t = 3
        self.model_lr_scheduler = CosineLRScheduler(
            self.model_optimizer,
            t_initial=24,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=1.0,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )
# def
#     num_epochs = cfg.MAX_EPOCH
#     lr_min = cfg.LR_MIN * cfg.BASE_LR
#     warmup_lr_init = cfg.LR_INIT * cfg.BASE_LR
#     warmup_t = cfg.WARMUP_EPOCHS
#
#     lr_scheduler = CosineLRScheduler(
#         optimizer,
#         t_initial=num_epochs,
#         lr_min=lr_min,
#         t_mul= 1.,
#         decay_rate=1.0,
#         warmup_lr_init=warmup_lr_init,
#         warmup_t=warmup_t,
#         cycle_limit=1,
#         t_in_epochs=True,
#         noise_range_t=None,
#         noise_pct= 0.67,
#         noise_std= 1.,
#         noise_seed=42,
#     )
#
#     return lr_scheduler
# def make_optimizer_2stage(cfg, model, center_criterion):
#     params = []
#     keys = []
#     for key, value in model.named_parameters():
#         if "text_encoder" in key:
#             value.requires_grad_(False)
#             continue
#         if "prompt_learner" in key:
#             value.requires_grad_(False)
#             continue
#         if not value.requires_grad:
#             continue
#         lr = cfg.SOLVER.STAGE2.BASE_LR
#         weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
#         if "bias" in key:
#             lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
#             weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
#         if cfg.SOLVER.STAGE2.LARGE_FC_LR:
#             if "classifier" in key or "arcface" in key:
#                 lr = cfg.SOLVER.BASE_LR * 2
#                 print('Using two times learning rate for fc ')
#
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#         keys += [key]
#     if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
#         optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
#     elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
#         optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
#     else:
#         optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
#     optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)
#
#     return optimizer, optimizer_center
    # def _init_optimizer(self):
    #
    #     rgb_model_params_group = [{'params': self.rgb_model.parameters(), 'lr': self.learning_rate,
    #                      'weight_decay': self.weight_decay}]
    #     ir_model_params_group = [{'params': self.ir_model.parameters(), 'lr': self.learning_rate,
    #                                'weight_decay': self.weight_decay}]
    #     shared_model_params_group = [{'params': self.shared_model.parameters(), 'lr': self.learning_rate,
    #                                'weight_decay': self.weight_decay}]
    #     classifier_params_group = [{'params': self.classifier.parameters(),
    #                                 'lr': self.c_learning_rate, 'weight_decay': self.weight_decay}]
    #
    #     self.rgb_model_optimizer = optim.Adam(rgb_model_params_group)
    #     self.rgb_model_lr_scheduler = WarmupMultiStepLR(self.rgb_model_optimizer, self.milestones,
    #                                          gamma=0.1, warmup_factor=0.01, warmup_iters=10)
    #
    #     self.ir_model_optimizer = optim.Adam(ir_model_params_group)
    #     self.ir_model_lr_scheduler = WarmupMultiStepLR(self.ir_model_optimizer, self.milestones,
    #                                                     gamma=0.1, warmup_factor=0.01, warmup_iters=10)
    #
    #     self.shared_model_optimizer = optim.Adam(shared_model_params_group)
    #     self.shared_model_lr_scheduler = WarmupMultiStepLR(self.shared_model_optimizer, self.milestones,
    #                                                    gamma=0.1, warmup_factor=0.01, warmup_iters=10)
    #
    #     self.classifier_optimizer = optim.Adam(classifier_params_group)
    #     self.classifier_lr_scheduler = WarmupMultiStepLR(self.classifier_optimizer, self.milestones,
    #                                                      gamma=0.1, warmup_factor=0.01, warmup_iters=10)


    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, 'best_model.pth'.format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)


        # if self.max_save_model_num > 0:
        #     root, _, files = os_walk(self.save_model_path)
        #     for file in files:
        #         if '.pth' not in file:
        #             files.remove(file)
        #     if len(files) > 4 * self.max_save_model_num:
        #         file_iters = sorted([int(file.replace('.pth', '').split('_')[1]) for file in files], reverse=False)
        #
        #         shared_model_file_path = os.path.join(root, 'model.pth'.format(file_iters[0]))
        #         os.remove(shared_model_file_path)




    def resume_last_model(self):
        root, _, files = os_walk(self.save_model_path)
        for file in files:
            if '.pth' not in file:
                files.remove(file)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pth', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1])
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_path):
        model_path = os.path.join(resume_path, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print('Successfully resume shared_model from {}'.format(model_path))


    def set_train(self):
        self.model = self.model.train()


        self.training = True

    def set_eval(self):
        self.model = self.model.eval()

        self.training = False

# class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
#                  warmup_method='linear', last_epoch=-1):
#         if not list(milestones) == sorted(milestones):
#             raise ValueError(
#                 "Milestones should be a list of " " increasing integers. Got {}", milestones)
#
#         if warmup_method not in ("constant", "linear"):
#             raise ValueError(
#                 "Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
#         self.milestones = milestones
#         self.gamma = gamma
#         self.warmup_factor = warmup_factor
#         self.warmup_iters = warmup_iters
#         self.warmup_method = warmup_method
#         super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         warmup_factor = 1
#         if self.last_epoch < self.warmup_iters:
#             if self.warmup_method == "constant":
#                 warmup_factor = self.warmup_factor
#             elif self.warmup_method == "linear":
#                 alpha = float(self.last_epoch) / float(self.warmup_iters)
#                 warmup_factor = self.warmup_factor * (1 - alpha) + alpha
#
#         return [
#             base_lr
#             * warmup_factor
#             * self.gamma ** bisect_right(self.milestones, self.last_epoch)
#             for base_lr in self.base_lrs
#         ]
