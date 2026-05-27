import torch
import torch.nn.functional as F
import copy
from torch.cuda import amp
from tools.loss import diversity_loss

def process_vedio0(x1,seq_len=6):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1

def process_vedio(x1,seq_len=6):
    b, s,c, h, w = x1.size()
    x1 = x1.view(b * s, c, h, w)
    return x1

def _parse_data_rgb(inputs):
    imgs,imgs1, _, pids, _, indexes = inputs
    return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

def _parse_data_ir(inputs):
    imgs, _, pids, _, indexes = inputs
    return imgs.cuda(), pids.cuda(), indexes.cuda()


def foward_video(iter,base,meter,scaler,epoch):
    # scaler = amp.GradScaler()
    for i in range(len(base.train_loader_ir)):
        inputs_ir = base.train_loader_ir.next_one()
        inputs_rgb = base.train_loader_rgb.next_one()

        # process inputs
        inputs_ir, labels_ir, indexes_ir = _parse_data_ir(inputs_ir)
        inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = _parse_data_rgb(inputs_rgb)
        inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
        labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
        indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)

        input_ir = process_vedio(inputs_ir,base.seq_lenth)
        input_rgb = process_vedio(inputs_rgb,base.seq_lenth)

        base.model_optimizer.zero_grad()

        with amp.autocast(enabled=True):
            # features, cls_score,prompt = base.model(x1=rgb_imgs, x2=ir_imgs)
            features,_,_ = base.model(x1=input_rgb, x2=input_ir)
            feature = features[3]
            feature_frame = features[1]
            
            # 对普通特征进行分割
            f_out_rgb,f_out_ir = feature.split([inputs_rgb.size(0),inputs_ir.size(0)],dim=0)
            f_out_rgb_frame,f_out_ir_frame = feature_frame.split([inputs_rgb.size(0),inputs_ir.size(0)],dim=0)
            
            
            
            # 计算loss
            loss_ir = base.memory_ir(f_out_ir, labels_ir)
            loss_rgb = base.memory_rgb(f_out_rgb, labels_rgb)
            # 使用新的标签计算帧级损失
            loss_ir_frame = base.memory_ir_frame(f_out_ir_frame, labels_ir)
            loss_rgb_frame = base.memory_rgb_frame(f_out_rgb_frame, labels_rgb)


            #邻域学习
            ir_rgb_loss = torch.tensor([0.]).cuda()
            rgb_ir_loss = torch.tensor([0.]).cuda()
            rgb_rgb_loss = torch.tensor([0.]).cuda()
            ir_ir_loss = torch.tensor([0.]).cuda()
            # 初始化帧级损失变量
            ir_rgb_loss_frame = torch.tensor([0.]).cuda()
            rgb_ir_loss_frame = torch.tensor([0.]).cuda()
            rgb_rgb_loss_frame = torch.tensor([0.]).cuda()
            ir_ir_loss_frame = torch.tensor([0.]).cuda()

            lamda_frame_neibor=0.5
            lamda_video_neibor=1
            # 邻域损失计算
            if epoch >= base.hm:  # 超过设定轮次才使用邻域损失
                ht_thresh = base.ht_thresh  # 阈值
                hm_thresh = base.hm_thresh  # 同模态阈值
                if epoch >= base.ht:
                    if epoch % 2 == 0:  # 奇偶轮次使用不同的邻域损失计算方式
                        # RGB->IR邻域损失(跨模态)
                        with torch.no_grad():
                            # 计算序列级相似度矩阵
                            sim_rgb_ir = base.memory_rgb_instance.features.detach()[indexes_rgb].mm(base.memory_ir_instance.features.detach().data.t())        
                            sim_frame_rgb_ir = base.memory_rgb_frame_instance.features.detach()[indexes_rgb].mm(base.memory_ir_frame_instance.features.detach().t())
                            
                            # 序列级和帧级的最近邻
                            nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
                            nearest_frame_rgb_ir = sim_frame_rgb_ir.max(dim=1, keepdim=True)[0]
                            
                            # 构建邻域掩码
                            mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * ht_thresh).detach().data
                            mask_neighbor_frame_rgb_ir = torch.gt(sim_frame_rgb_ir, nearest_frame_rgb_ir * ht_thresh).detach().data
                            
                            num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_frame_rgb_ir).sum(dim=1) + 1
                        
                        # 计算RGB-IR损失
                        sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(base.memory_ir_instance.features.detach().data.t())
                        sim_rgb_ir_exp = sim_rgb_ir / 0.05
                        score_intra_rgb_ir = F.softmax(sim_rgb_ir_exp, dim=1)
                        score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
                        
                        rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_frame_rgb_ir).sum(dim=1)
                        rgb_ir_loss = lamda_video_neibor*rgb_ir_loss.div(num_neighbor_rgb_ir).mean()
                        
                        # 新增：帧级损失计算
                        sim_frame_rgb_ir_exp = sim_frame_rgb_ir / 0.05
                        score_intra_frame_rgb_ir = F.softmax(sim_frame_rgb_ir_exp, dim=1)
                        score_intra_frame_rgb_ir = score_intra_frame_rgb_ir.clamp_min(1e-8)
                        rgb_ir_loss_frame = -score_intra_frame_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_frame_rgb_ir).sum(dim=1)
                        rgb_ir_loss_frame = lamda_frame_neibor*rgb_ir_loss_frame.div(num_neighbor_rgb_ir).mean()
                        
                    else:
                        # IR->RGB邻域损失(跨模态)
                        with torch.no_grad():
                            # 计算序列级相似度矩阵
                            sim_ir_rgb = base.memory_ir_instance.features.detach()[indexes_ir].mm(base.memory_rgb_instance.features.detach().data.t())
                            
                            sim_frame_ir_rgb = base.memory_ir_frame_instance.features.detach()[indexes_ir].mm(base.memory_rgb_frame_instance.features.detach().t())
                            
                            # 序列级和帧级的最近邻
                            nearest_ir_rgb = sim_ir_rgb.max(dim=1, keepdim=True)[0]
                            nearest_frame_ir_rgb = sim_frame_ir_rgb.max(dim=1, keepdim=True)[0]
                            
                            # 构建邻域掩码
                            mask_neighbor_ir_rgb = torch.gt(sim_ir_rgb, nearest_ir_rgb * ht_thresh).detach().data
                            mask_neighbor_frame_ir_rgb = torch.gt(sim_frame_ir_rgb, nearest_frame_ir_rgb * ht_thresh).detach().data
                            
                            num_neighbor_ir_rgb = mask_neighbor_ir_rgb.mul(mask_neighbor_frame_ir_rgb).sum(dim=1) + 1
                        
                        # 计算IR-RGB损失
                        sim_ir_rgb = F.normalize(f_out_ir, dim=1).mm(base.memory_rgb_instance.features.detach().data.t())
                        sim_ir_rgb_exp = sim_ir_rgb / 0.05
                        score_intra_ir_rgb = F.softmax(sim_ir_rgb_exp, dim=1)
                        score_intra_ir_rgb = score_intra_ir_rgb.clamp_min(1e-8)
                        
                        ir_rgb_loss = -score_intra_ir_rgb.log().mul(mask_neighbor_ir_rgb).mul(mask_neighbor_frame_ir_rgb).sum(dim=1)
                        ir_rgb_loss = lamda_video_neibor*ir_rgb_loss.div(num_neighbor_ir_rgb).mean()
                        
                        # 新增：帧级损失计算
                        sim_frame_ir_rgb_exp = sim_frame_ir_rgb / 0.05
                        score_intra_frame_ir_rgb = F.softmax(sim_frame_ir_rgb_exp, dim=1)
                        score_intra_frame_ir_rgb = score_intra_frame_ir_rgb.clamp_min(1e-8)
                        ir_rgb_loss_frame = -score_intra_frame_ir_rgb.log().mul(mask_neighbor_ir_rgb).mul(mask_neighbor_frame_ir_rgb).sum(dim=1)
                        ir_rgb_loss_frame = lamda_frame_neibor*ir_rgb_loss_frame.div(num_neighbor_ir_rgb).mean()
                    
                # RGB-RGB邻域损失(同模态)
                with torch.no_grad():
                    sim_rgb_rgb = base.memory_rgb_instance.features.detach()[indexes_rgb].mm(base.memory_rgb_instance.features.detach().data.t())
                    sim_frame_rgb_rgb = base.memory_rgb_frame_instance.features.detach()[indexes_rgb].mm(base.memory_rgb_frame_instance.features.detach().data.t())
                    
                    # 序列级和帧级的最近邻
                    nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
                    nearest_frame_rgb_rgb = sim_frame_rgb_rgb.max(dim=1, keepdim=True)[0]
                    
                    # 构建邻域掩码
                    mask_neighbor_rgb_rgb = torch.gt(sim_rgb_rgb, nearest_rgb_rgb * hm_thresh).detach().data
                    mask_neighbor_frame_rgb_rgb = torch.gt(sim_frame_rgb_rgb, nearest_frame_rgb_rgb * hm_thresh).detach().data

                    num_neighbor_rgb_rgb = mask_neighbor_rgb_rgb.mul(mask_neighbor_frame_rgb_rgb).sum(dim=1)+1

                
                sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(base.memory_rgb_instance.features.detach().data.t())
                sim_rgb_rgb_exp = sim_rgb_rgb / 0.05
                score_intra_rgb_rgb = F.softmax(sim_rgb_rgb_exp, dim=1)
                score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
                
                rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_frame_rgb_rgb).sum(dim=1)
                rgb_rgb_loss = lamda_video_neibor*rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()
                
                # 新增：帧级损失计算
                sim_frame_rgb_rgb_exp = sim_frame_rgb_rgb / 0.05  # 序列数量*总序列数量
                score_intra_frame_rgb_rgb = F.softmax(sim_frame_rgb_rgb_exp, dim=1)
                score_intra_frame_rgb_rgb = score_intra_frame_rgb_rgb.clamp_min(1e-8)
                rgb_rgb_loss_frame = -score_intra_frame_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_frame_rgb_rgb).sum(dim=1)
                rgb_rgb_loss_frame = lamda_frame_neibor*rgb_rgb_loss_frame.div(num_neighbor_rgb_rgb).mean()
                
                # IR-IR邻域损失(同模态)
                with torch.no_grad():
                    sim_ir_ir = base.memory_ir_instance.features.detach()[indexes_ir].mm(base.memory_ir_instance.features.detach().data.t())
                    sim_frame_ir_ir = base.memory_ir_frame_instance.features.detach()[indexes_ir].mm(base.memory_ir_frame_instance.features.detach().data.t())
                    
                    # 序列级和帧级的最近邻
                    nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
                    nearest_frame_ir_ir = sim_frame_ir_ir.max(dim=1, keepdim=True)[0]
                    
                    # 构建邻域掩码
                    mask_neighbor_ir_ir = torch.gt(sim_ir_ir, nearest_ir_ir * hm_thresh).detach().data
                    mask_neighbor_frame_ir_ir = torch.gt(sim_frame_ir_ir, nearest_frame_ir_ir * hm_thresh).detach().data

                    num_neighbor_ir_ir = mask_neighbor_ir_ir.mul(mask_neighbor_frame_ir_ir).sum(dim=1)+1
                
                sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(base.memory_ir_instance.features.detach().data.t())
                sim_ir_ir_exp = sim_ir_ir / 0.05
                score_intra_ir_ir = F.softmax(sim_ir_ir_exp, dim=1)
                score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
                
                ir_ir_loss = -score_intra_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_frame_ir_ir).sum(dim=1)
                ir_ir_loss = lamda_video_neibor*ir_ir_loss.div(num_neighbor_ir_ir).mean()

                sim_frame_ir_ir_exp =sim_frame_ir_ir /0.05  # 64*13638
                score_intra_frame_ir_ir =  F.softmax(sim_frame_ir_ir_exp,dim=1)
                score_intra_frame_ir_ir = score_intra_frame_ir_ir.clamp_min(1e-8)
                ir_ir_loss_frame = -score_intra_frame_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_frame_ir_ir).sum(dim=1)
                ir_ir_loss_frame = lamda_frame_neibor*ir_ir_loss_frame.div(num_neighbor_ir_ir).mean()

        if epoch >=base.ht:
            loss=(loss_ir + loss_rgb + loss_ir_frame + loss_rgb_frame)+0.1*(ir_ir_loss+rgb_rgb_loss+ir_ir_loss_frame+rgb_rgb_loss_frame)+0.1*(rgb_ir_loss+ir_rgb_loss+rgb_ir_loss_frame+ir_rgb_loss_frame)#+0*(loss_ins_ir+loss_ins_rgb+loss_ins_ir_s+loss_ins_rgb_s)#+0.1*(loss_camera_ir+loss_camera_rgb)
        else:
        # loss=loss_rgb_s+loss_ir_s+loss_ir+loss_rgb+0.5*(ir_ir_loss+rgb_rgb_loss+ir_ir_loss_s+rgb_rgb_loss_s)+0.5*(rgb_ir_loss+ir_rgb_loss+rgb_ir_loss_s+ir_rgb_loss_s)#+0*(loss_ins_ir+loss_ins_rgb+loss_ins_ir_s+loss_ins_rgb_s)#+0.1*(loss_camera_ir+loss_camera_rgb)
            loss=(loss_ir + loss_rgb + loss_ir_frame + loss_rgb_frame)+1*(ir_ir_loss+rgb_rgb_loss+ir_ir_loss_frame+rgb_rgb_loss_frame)+0.5*(rgb_ir_loss+ir_rgb_loss+rgb_ir_loss_frame+ir_rgb_loss_frame)#+0*(loss_ins_ir+loss_ins_rgb+loss_ins_ir_s+loss_ins_rgb_s)#+0.1*(loss_camera_ir+loss_camera_rgb)


            
        with torch.no_grad():
            base.memory_ir_instance.updateEM(f_out_ir, indexes_ir)
            base.memory_rgb_instance.updateEM(f_out_rgb, indexes_rgb)
            base.memory_ir_frame_instance.updateEM(f_out_ir_frame, indexes_ir)
            base.memory_rgb_frame_instance.updateEM(f_out_rgb_frame, indexes_rgb)
        
        
        scaler.scale(loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()

        meter.update({'loss_ir': loss_ir.data,
                      'loss_rgb': loss_rgb.data,
                      'loss_ir_frame': loss_ir_frame.data,
                      'loss_rgb_frame': loss_rgb_frame.data,
                      'loss_neibor_rgb_rgb':rgb_rgb_loss,
                      'loss_neibor_rgb_rgb_frame':rgb_rgb_loss_frame,
                      'loss_neibor_ir_ir':ir_ir_loss,
                      'loss_neibor_ir_ir_frame':ir_ir_loss_frame,
                      'loss_neibor_ir_rgb':ir_rgb_loss,
                      'loss_neibor_ir_rgb_frame':ir_rgb_loss_frame,                      
                      'loss_neibor_rgb_ir':rgb_ir_loss,
                      'loss_neibor_rgb_ir_frame':rgb_ir_loss_frame,
                      'total_loss': loss.data
                      })
    return meter

def foward_video_2(iter,base,meter,scaler,epoch):
    # scaler = amp.GradScaler()
    for i in range(len(base.train_loader_ir)):
        inputs_ir = base.train_loader_ir.next_one()
        inputs_rgb = base.train_loader_rgb.next_one()

        # process inputs
        inputs_ir, labels_ir, indexes_ir = _parse_data_ir(inputs_ir)
        inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = _parse_data_rgb(inputs_rgb)
        inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
        labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)

        input_ir = process_vedio(inputs_ir,base.seq_lenth)
        input_rgb = process_vedio(inputs_rgb,base.seq_lenth)

        base.model_optimizer.zero_grad()

        with amp.autocast(enabled=True):
            # features, cls_score,prompt = base.model(x1=rgb_imgs, x2=ir_imgs)
            features,_,_ = base.model(x1=input_rgb, x2=input_ir)
            feature = features[3]
            feature_frame = features[1]
            
            # 对普通特征进行分割
            f_out_rgb,f_out_ir = feature.split([inputs_rgb.size(0),inputs_ir.size(0)],dim=0)
            f_out_rgb_frame,f_out_ir_frame = feature_frame.split([inputs_rgb.size(0),inputs_ir.size(0)],dim=0)
            
            
            
            # 计算loss
            loss_ir = base.memory_ir(f_out_ir, labels_ir)
            loss_rgb = base.memory_rgb(f_out_rgb, labels_rgb)
            # 使用新的标签计算帧级损失
            loss_ir_frame = base.memory_ir_frame(f_out_ir_frame, labels_ir)
            loss_rgb_frame = base.memory_rgb_frame(f_out_rgb_frame, labels_rgb)
            
            total_loss = loss_ir + loss_rgb + loss_ir_frame + loss_rgb_frame

        scaler.scale(total_loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()

        meter.update({'loss_ir': loss_ir.data,
                      'loss_rgb': loss_rgb.data,
                      'loss_ir_frame': loss_ir_frame.data,
                      'loss_rgb_frame': loss_rgb_frame.data,
                      'total_loss': total_loss.data
                      })
    return meter