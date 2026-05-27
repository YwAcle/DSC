
import numpy as np
import torch
from torch.autograd import Variable
from tools import eval_regdb, eval_sysu,evaluate_vcm,MODALITY_,evaluate_bupt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test(base, loader, config):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    query_feat = np.zeros((loader.n_query, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader.query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = base.ir_model(input)
            feat = base.shared_model(feat)
            feat = base.classifier(feat)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    print('Extracting Gallery Feature...')

    if loader.dataset == 'sysu':
        all_cmc = 0
        all_mAP = 0
        all_mINP = 0
        for i in range(10):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            gall_feat = np.zeros((loader.n_gallery, 768))
            with torch.no_grad():
                for batch_idx, (input, label) in enumerate(gall_loader):
                    batch_num = input.size(0)
                    input = Variable(input.cuda())
                    feat = base.rgb_model(input)
                    feat = base.shared_model(feat)
                    feat = base.classifier(feat)
                    gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

                    ptr = ptr + batch_num
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            cmc, mAP, mINP = eval_sysu(-distmat, loader.query_label, loader.gall_label, loader.query_cam,
                                       loader.gall_cam)
            all_cmc += cmc
            all_mAP += mAP
            all_mINP += mINP
        all_cmc /= 10.0
        all_mAP /= 10.0
        all_mINP /= 10.0

    elif loader.dataset == 'regdb':
        gall_loader = loader.gallery_loaders
        gall_feat = np.zeros((loader.n_gallery, 768))
        ptr = 0
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat = base.rgb_model(input)
                feat = base.shared_model(feat)
                feat = base.classifier(feat)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

                ptr = ptr + batch_num
        if config.regdb_test_mode == 't-v':
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            cmc, mAP, mINP = eval_regdb(-distmat, loader.query_label, loader.gall_label)
        else:
            distmat = np.matmul(gall_feat, np.transpose(query_feat))
            cmc, mAP, mINP = eval_regdb(-distmat, loader.gall_label, loader.query_label)

        all_cmc, all_mAP, all_mINP = cmc, mAP, mINP


    return all_cmc, all_mAP, all_mINP


def process_vedio(x1,seq_len=6):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1

def tem_pool(features,t=6):
    features = features.squeeze()
    features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
    features = features.mean(0)
    return features


def test_vcm(base, loader, t2v=True):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    q_pids = []
    q_camids = []

    # if config.vcm_test_mode == 't2v':
    #     t2v = True
    # else:
    #     t2v = False

    if t2v:
        query_loader = loader.query_loader
        query_feat = np.zeros((loader.samples.num_query_tracklets, 1536))
    else:
        query_loader = loader.query_loader_1
        query_feat = np.zeros((loader.samples.num_query_tracklets_1, 1536))

    with torch.no_grad():
        for batch_idx, (input, label,c_label,img_path1,img_path2,img_path3) in enumerate(query_loader):
            batch_num = input.size(0)
            q_pids.extend(label)
            q_camids.extend(c_label)
            input = Variable(input.cuda())
            input = process_vedio(input)
            if t2v:
                feat = base.model(x2=input)
            else:
                feat = base.model(x1=input)
            query = feat
            
            query_feat[ptr:ptr + batch_num, :] = query.detach().cpu().numpy()
            ptr = ptr + batch_num
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
    print('Extracting Gallery Feature...')

    if loader.dataset == 'vcm':
        g_pids = []
        g_camids = []

        ptr = 0
        if t2v:
            gall_loader = loader.gallery_loader
            gall_feat = np.zeros((loader.samples.num_gallery_tracklets, 1536))
        else:
            gall_loader = loader.gallery_loader_1
            gall_feat = np.zeros((loader.samples.num_gallery_tracklets_1, 1536))

        _datasets = [ query_loader.dataset.dataset,gall_loader.dataset.dataset,]
        with torch.no_grad():
            for batch_idx, (input, label,c_label,img_path1,img_path2,img_path3) in enumerate(gall_loader):
                batch_num = input.size(0)
                g_pids.extend(label)
                g_camids.extend(c_label)
                input = Variable(input.cuda())
                input = process_vedio(input)
                if t2v:
                    feat = base.model(x1=input)
                else:
                    feat = base.model(x1=input)

                gall = feat
                
                gall_feat[ptr:ptr + batch_num, :] = gall.detach().cpu().numpy()
                ptr = ptr + batch_num

        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        # cmc2, mAP2 = evaluate_vcm(-distmat, q_pids, g_pids, q_camids,
        #                           g_camids)
        visualize_output_path = 'vcm_ir2rgb_retrieval'
        # visualize_ranked_results2(distmat, _datasets, visualize_output_path)
        visualize_ranked_results(distmat,_datasets,visualize_output_path)
        # all_cmc  = cmc2
        # all_mAP  = mAP2

    #     if t2v:
    #         str = print_metrics(
    #             all_cmc, all_mAP,
    #             prefix='{:<3}->{:<3}:  '.format('IR', 'RGB')
    #         )
    #     else:
    #         str = print_metrics(
    #             all_cmc, all_mAP,
    #             prefix='{:<3}->{:<3}:  '.format('RGB', 'IR')
    #         )
    # return all_cmc, all_mAP, str


def print_metrics(cmc, ap, prefix=''):
    str = '{}mAP: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}.'.format(prefix, ap, cmc[0], cmc[4], cmc[9], cmc[19])
    return str

def test_bupt(base, loader, config):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    q_pids = []
    q_camids = []
    q_modalitys = []
    query_feats = []
    if config.vcm_test_mode == 't2v':
        t2v = True
    else:
        t2v = False


    query_loader = loader.query_loader
    query_feat = np.zeros((1048, 768))


    with torch.no_grad():
        for batch_idx, (input, label,c_label,modals) in enumerate(query_loader):
            batch_num = input.size(0)
            q_pids.extend(label)
            q_camids.extend(c_label)
            q_modalitys.extend(modals)
            input = Variable(input.cuda())
            input = process_vedio(input)
            if modals[0]==0:
                feat = base.model(x1=input)
            elif modals[0]==1:
                feat = base.model(x2=input)
            else:
                continue
            # features_map,features = base.shared_model(feat)
            # features = tem_pool(features)

            # feat = base.classifier(features)
            query = feat

            query_feat[ptr:ptr + batch_num, :] = query.detach().cpu().numpy()
            ptr = ptr + batch_num
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_modalitys = np.asarray(q_modalitys)
        # query_feat = torch.cat(query_feats, dim=0)
    print('Extracting Gallery Feature...')

    if loader.dataset == 'bupt':
        g_pids = []
        g_camids = []
        g_modalitys = []
        # gall_feats = []
        ptr = 0

        gall_loader = loader.gallery_loader
        gall_feat = np.zeros((4790, 768))


        with torch.no_grad():
            for batch_idx, (input, label,c_label,modals) in enumerate(gall_loader):
                batch_num = input.size(0)
                g_pids.extend(label)
                g_camids.extend(c_label)
                g_modalitys.extend(modals)
                input = Variable(input.cuda())
                input = process_vedio(input)
                if modals[0]==0:
                    feat = base.model(x1=input)
                elif modals[1]==1:
                    feat = base.model(x2=input)
                else:
                    continue
                gall = feat
                gall_feat[ptr:ptr + batch_num, :] = gall.detach().cpu().numpy()
                # gall_feats.append(gall.detach().cpu().numpy())
                ptr = ptr + batch_num
        # gall_feat = torch.cat(gall_feats, dim=0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_modalitys = np.asarray(g_modalitys)
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        # evaluate (intra/inter-modality)
        CMC, MAP = [], []
        eval_str = []
        for q_modal in (0, 1):
            for g_modal in (0, 1):
                if q_modal!=g_modal:
                    q_mask = q_modalitys == q_modal
                    g_mask = g_modalitys == g_modal
                    tmp_distance = distmat[q_mask, :][:, g_mask]
                    tmp_qid = q_pids[q_mask]
                    tmp_gid = g_pids[g_mask]
                    # tem_qcid = q_camids[q_mask]
                    # tem_gcid = g_camids[g_mask]
                    tmp_cmc, tmp_ap = evaluate_bupt(-tmp_distance, tmp_qid, tmp_gid, config)
                    CMC.append(tmp_cmc * 100)
                    MAP.append(tmp_ap * 100)

                    str = print_metrics(
                        tmp_cmc, tmp_ap,
                        prefix='{:<3}->{:<3}:  '.format(MODALITY_[q_modal], MODALITY_[g_modal])
                    )
                    eval_str.append(str)




    #     cmc2, mAP2 = evaluate_vcm(-distmat, q_pids, g_pids, q_camids,
    #                               g_camids)
    #
    #     all_cmc  = cmc2
    #     all_mAP  = mAP2
    #     all_mINP = mAP2
    #
    return CMC, MAP,eval_str

import numpy as np
import os
import os.path as osp
import shutil
import sys
from PIL import Image, ImageOps, ImageDraw

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))


def visualize_ranked_results2(distmat, dataset, save_dir='', topk=6):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)[:, ::-1]
    make_dirs(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            make_dirs(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path))
        make_dirs(qdir)
        _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1
                if rank_idx > topk:
                    break

    print("Done")

def concat_images(images):
    """
    将6张图片拼接成一张大图，上面3张，下面3张。

    参数:
    images -- 包含6个图像对象的列表

    返回:
    拼接后的图像对象
    """
    if len(images) != 6:
        raise ValueError("输入的图片列表必须包含6张图片")

    # 获取每张图像的宽度和高度
    widths, heights = zip(*(i.size for i in images))

    # 计算总宽度和总高度
    total_width = sum(widths[:3])  # 第一行的总宽度
    total_height = sum(heights[3:])  # 第二行的总高度

    # 创建一个新的图像，尺寸为 3 张宽度 * 2 行高度
    new_im = Image.new('RGB', (total_width, max(heights[:3]) + max(heights[3:])))

    # 将图片粘贴到新图像中
    x_offset = 0
    for i in range(3):  # 上面3张图像
        new_im.paste(images[i], (x_offset, 0))  # 第一行，粘贴图像
        x_offset += images[i].width  # 更新偏移量

    y_offset = max(heights[:3])  # 第二行的y_offset从第一行的最大高度开始
    x_offset = 0
    for i in range(3, 6):  # 下面3张图像
        new_im.paste(images[i], (x_offset, y_offset))  # 第二行，粘贴图像
        x_offset += images[i].width  # 更新偏移量

    return new_im

def visualize_ranked_results(distmat, dataset, save_dir='', topk=10, sort='descend', mode='inter-camera', only_show=None):
    """Visualizes ranked results.
    Args:
        dismat (numpy.ndarray): distance matrix of shape (nq, ng)
        dataset (tupple): a 2-tuple including (query,gallery), each of which contains
            tuples of (img_paths, pids, camids)
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
        sort (string): ascend means small value is similar, otherwise descend
        mode (string): intra-camera/inter-camera/all
            intra-camera only visualize results in the same camera with the query
            inter-camera only visualize results in the different camera with the query
            all visualize all results
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    assert sort in ['descend', 'ascend']
    assert mode in ['intra-camera', 'inter-camera', 'all']

    if sort is 'ascend':
        indices = np.argsort(distmat, axis=1)
    elif sort is 'descend':
        indices = np.argsort(distmat, axis=1)[:, ::-1]

    make_dirs(save_dir)




    def cat_imgs_to(image_list, hit_list, text_list, target_dir):

        images = []
        for img, hit, text in zip(image_list, hit_list, text_list):
            imgs = []
            for i in range(6):
                img_i = img[i]
                img_i = Image.open(img_i).resize((64, 128))
                imgs.append(img_i)
            img = concat_images(imgs)
            if hit:
                img = ImageOps.expand(img, border=4, fill='green')
            else:
                img = ImageOps.expand(img, border=4, fill='red')
            # img = Image.open(img).resize((64, 128))

            # d = ImageDraw.Draw(img)
            # d.text((3, 1), "{:.3}".format(text), fill=(255, 255, 0))

            images.append(img)

        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width+110, max_height),(255, 255, 255))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += 10
            x_offset += im.size[0]

        new_im.save(target_dir)

    counts = 0
    for q_idx in range(num_q):

        image_list = []
        hit_list = []
        text_list = []

        # query image
        qimg_path, qpid, qcamid = query[q_idx]
        # qimg_path = qimg_paths[0]
        image_list.append(qimg_path)
        hit_list.append(True)
        text_list.append(0.0)

        # target dir
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path))


        # matched images
        rank_idx = 1
        for ii, g_idx in enumerate(indices[q_idx, :]):
            gimg_path, gpid, gcamid = gallery[g_idx]
            # gimg_path = gimg_paths[0]
            if mode == 'intra-camera':
                valid = qcamid == gcamid
            elif mode == 'inter-camera':
                valid = (qpid != gpid and qcamid == gcamid) or (qcamid != gcamid)
            elif mode == 'all':
                valid = True
            if valid:
                if only_show == 'pos' and qpid != gpid: continue
                if only_show == 'neg' and qpid == gpid: continue
                image_list.append(gimg_path)
                hit_list.append(qpid == gpid)
                text_list.append(distmat[q_idx, g_idx])
                rank_idx += 1
                if rank_idx > topk:
                    break

        counts += 1
        path = os.path.dirname(qdir)
        img_path = os.path.join(path,str(counts)+'.jpg')
        cat_imgs_to(image_list, hit_list, text_list, img_path)
        print(counts, qdir)
