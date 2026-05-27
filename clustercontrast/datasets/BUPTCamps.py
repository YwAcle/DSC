import json
import os.path as osp


def relabel(meta_data):
    pid_set = set()
    for _, pid, _ in meta_data:
        pid_set.add(pid)
    pid2label = {pid: label for label, pid in enumerate(list(pid_set))}

    relabel_data = []
    for i, data in enumerate(meta_data):
        label = pid2label[data[1]]
        relabel_data.append((data[0], label, data[2]))

    return relabel_data


def count_id(meta_data):
    pid_set = set()
    for _, pid, _ in meta_data:
        pid_set.add(pid)
    return len(pid_set)


def spilt_dense_data(tracklets, chunk_size=10):
    new_tracklets = []

    # 原始长序列
    for frm_paths, pid, camid in tracklets:
        frm_num = len(frm_paths)

        # 重组短序列
        for i in range(0, frm_num, chunk_size):
            if i + chunk_size > frm_num: break
            new_frm_paths = frm_paths[i:i + chunk_size]
            new_tracklets.append((new_frm_paths, pid, camid))
    return new_tracklets


class BUPTCampus_Dataset():
    def __init__(self, root, trial=0, min_seq_len=12) -> None:
        self.data_root = osp.join(root, 'DATA')
        self.images_dir = self.data_root
        self.data_paths = json.load(open(osp.join(root, 'data_paths.json')))

        data_info_train = self._parse_data(osp.join(root, 'train.txt'))
        data_info_query = self._parse_data(osp.join(root, 'query.txt'))
        data_info_gallery = self._parse_data(osp.join(root, 'gallery.txt'))

        self.train_ir, self.train_rgb = self._process_train(data_info_train)
        self.query_ir, self.query_rgb = self._process_test(data_info_query)
        self.gallery_ir, self.gallery_rgb = self._process_test(data_info_gallery)

        self.num_train_ir_pids = count_id(self.train_ir)
        self.num_train_rgb_pids = count_id(self.train_rgb)
        self.num_query_ir_pids = count_id(self.query_ir)
        self.num_query_rgb_pids = count_id(self.query_rgb)
        self.num_gallery_ir_pids = count_id(self.gallery_ir)
        self.num_gallery_rgb_pids = count_id(self.gallery_rgb)
        
        self.num_train_ir_tracklets = len(self.train_ir)
        self.num_train_rgb_tracklets = len(self.train_rgb)
        self.num_query_ir_tracklets = len(self.query_ir)
        self.num_query_rgb_tracklets = len(self.query_rgb)
        self.num_gallery_ir_tracklets = len(self.gallery_ir)
        self.num_gallery_rgb_tracklets = len(self.gallery_rgb)

        self._verbose()

    def _verbose(self):
        print("\n=> BUPTCampus loaded")
        print("Dataset statistics:")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("train_ir    | {:5d} | {:8d}".format(self.num_train_ir_pids,   self.num_train_ir_tracklets))
        print("train_rgb   | {:5d} | {:8d}".format(self.num_train_rgb_pids,  self.num_train_rgb_tracklets))
        print("query_ir    | {:5d} | {:8d}".format(self.num_query_ir_pids,   self.num_query_ir_tracklets))
        print("gallery_rgb | {:5d} | {:8d}".format(self.num_gallery_rgb_pids,self.num_gallery_rgb_tracklets))
        print("query_rgb   | {:5d} | {:8d}".format(self.num_query_rgb_pids,  self.num_query_rgb_tracklets))
        print("gallery_ir  | {:5d} | {:8d}".format(self.num_gallery_ir_pids, self.num_gallery_ir_tracklets))
        print("---------------------------------")

    def _parse_data(self, path):
        data_info = []
        path = osp.join(self.data_root, path)
        with open(path) as f:
            for line in f.readlines():
                obj_id, modality, camera, tracklet_id = line.strip().split(' ')
                data_info.append((obj_id, modality, camera, tracklet_id))
        return data_info
    
    def _process_train(self, data_info):
        tracklets_ir = []
        tracklets_rgb = []
        for i, data in enumerate(data_info):
            obj_id, modality, camera, tracklet_id = data

            data_paths_ir = self.data_paths[obj_id]['IR'][camera][tracklet_id]
            data_paths_ir = [osp.join(self.data_root, img_name) for img_name in data_paths_ir]
            tracklets_ir.append((data_paths_ir, obj_id, camera))
            
            data_paths_rgb = self.data_paths[obj_id]['RGB'][camera][tracklet_id]
            data_paths_rgb = [osp.join(self.data_root, img_name) for img_name in data_paths_rgb]
            tracklets_rgb.append((data_paths_rgb, obj_id, camera))

        # 拆分数据集
        tracklets_ir = spilt_dense_data(tracklets_ir, chunk_size=40)
        tracklets_rgb = spilt_dense_data(tracklets_rgb, chunk_size=40)
            
        tracklets_ir = relabel(tracklets_ir)
        tracklets_rgb = relabel(tracklets_rgb)
        return tracklets_ir, tracklets_rgb

    def  _process_test(self, data_info):
        """与VCM数据集不同，两种模态的query和gallery不共享数据"""
        tracklets_ir = []
        tracklets_rgb = []
        for i, data in enumerate(data_info):
            obj_id, modality, camera, tracklet_id = data
            if modality == 'IR':
                data_paths_ir = self.data_paths[obj_id][modality][camera][tracklet_id]
                data_paths_ir = [osp.join(self.data_root, img_name) for img_name in data_paths_ir]
                tracklets_ir.append((data_paths_ir, int(obj_id), camera))
            elif modality == 'RGB':
                data_paths_rgb = self.data_paths[obj_id][modality][camera][tracklet_id]
                data_paths_rgb = [osp.join(self.data_root, img_name) for img_name in data_paths_rgb]
                tracklets_rgb.append((data_paths_rgb, int(obj_id), camera))
        return tracklets_ir, tracklets_rgb


if __name__ == '__main__':
    dataset = BUPTCampus_Dataset('/data/wangsy/dataset/BUPTCampus/DATA')