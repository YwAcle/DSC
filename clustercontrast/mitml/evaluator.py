from __future__ import print_function
import time
import numpy as np
import torch
import os
from torch.autograd import Variable
from .eval_metrics import evaluate
from ..net import ForwardMode


class Tester():
    def __init__(self, model, dataloader, test_log_dir='') -> None:
        self.model = model
        self.dataloader = dataloader
        self.test_log_dir = test_log_dir

        self.best_cmc_v2t = 0
        self.best_cmc_t2v = 0
        self.best_mAP_v2t = 0
        self.best_mAP_t2v = 0

    def test(self, epoch):    
        vcm_test_loader = self.dataloader
        n_testset_rgb = vcm_test_loader.n_testset_rgb
        n_testset_ir = vcm_test_loader.n_testset_ir
        testloader_rgb = vcm_test_loader.testloader_rgb
        testloader_ir = vcm_test_loader.testloader_ir

        def _extract_features(model, dataloader, n_feat=0, mode=1):
            s = vcm_test_loader.seq_length
            model.eval()
            start = time.time()
            ptr = 0
            feat_ = np.zeros((n_feat, 2048))
            pids_, camids_ = [], []

            with torch.no_grad():
                for batch_idx, (imgs, pids, camids) in enumerate(dataloader):
                    input = imgs
                    batch_num = input.size(0)
                    b, sc, h, w = input.size()
                    input = input.view(b*s, sc//s, h, w)
                    input = Variable(input.cuda())
                    feat = model(input,input, mode=mode)
                    
                    # feat = feat.view(b, s, -1) # mitml中有时序处理模块，不要额外处理
                    # feat = torch.mean(feat, 1)
                    # feat = torch.max(feat, 1)[0]

                    feat_[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    ptr = ptr + batch_num
                    pids_.extend(pids)
                    camids_.extend(camids)

            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            time_cost = time.time() - start

            return feat_, pids_, camids_, time_cost

        print(f'Test Epoch: [{epoch}] Extracting Test Set Feature(RGB)...')
        feat_rgb, pids_rgb, camids_rgb, time_cost = \
            _extract_features(self.model, testloader_rgb, n_feat=n_testset_rgb, mode=1)
        print('Test Epoch: [{}] Extracting Time:\t {:.3f}'.format(epoch, time_cost))

        print(f'Test Epoch: [{epoch}] Extracting Test Set Feature(IR)...')
        feat_ir, pids_ir, camids_ir, time_cost = \
            _extract_features(self.model, testloader_ir, n_feat=n_testset_ir, mode=2)
        print('Test Epoch: [{}] Extracting Time:\t {:.3f}'.format(epoch, time_cost))


        print(f"Test Epoch: [{epoch}] Computing CMC and mAP")

        distmat_v2i = np.matmul(feat_rgb, np.transpose(feat_ir))
        cmc_v2t, mAP_v2t = evaluate(-distmat_v2i, pids_rgb, pids_ir, camids_rgb, camids_ir) 
        print("Test Epoch: [{}] FC('v2t') Rank-1: {:.2%}\t\tRank-5: {:.2%}\t\tRank-10: {:.2%}\t\tRank-20: {:.2%}\t\tmAP: {:.2%}"
                .format(epoch, cmc_v2t[0], cmc_v2t[4], cmc_v2t[9], cmc_v2t[19], mAP_v2t))
        
        if cmc_v2t[0] > self.best_cmc_v2t:
            self.best_cmc_v2t = cmc_v2t[0]
            state = {'model': self.model.state_dict(), 'mAP': self.best_cmc_v2t, 'epoch': epoch}
            # log_dir = './logs/vcm_s1'
            # torch.save(state, os.path.join(log_dir, 'v2t_rank1_best.pt'))
            torch.save(state, self.test_log_dir + '/v2t_rank1_best.pt')
        # if mAP_v2t > self.best_mAP_v2t:
        #     self.best_mAP_v2t = mAP_v2t
        #     state = {'model': self.model.state_dict(), 'mAP': self.best_mAP_v2t, 'epoch': epoch}
        #     torch.save(state, self.test_log_dir + 'v2t_mAP_best.t')

        distmat_i2v = np.matmul(feat_ir, np.transpose(feat_rgb))
        cmc_t2v, mAP_t2v = evaluate(-distmat_i2v, pids_ir, pids_rgb, camids_ir, camids_rgb) 
        print("Test Epoch: [{}] FC('t2v') Rank-1: {:.2%}\t\tRank-5: {:.2%}\t\tRank-10: {:.2%}\t\tRank-20: {:.2%}\t\tmAP: {:.2%}"
                .format(epoch, cmc_t2v[0], cmc_t2v[4], cmc_t2v[9], cmc_t2v[19], mAP_t2v))
        
        # if cmc_t2v[0] > self.best_cmc_t2v:
        #     self.best_cmc_t2v = cmc_t2v[0]
        #     state = {'model': self.model.state_dict(), 'mAP': self.best_cmc_t2v, 'epoch': epoch}
        #     torch.save(state, self.test_log_dir + 't2v_rank1_best.t')
        # if mAP_t2v > self.best_mAP_t2v:
        #     self.best_mAP_t2v = mAP_t2v
        #     state = {'model': self.model.state_dict(), 'mAP': self.best_mAP_t2v, 'epoch': epoch}
        #     torch.save(state, self.test_log_dir + 't2v_mAP_best.t')
