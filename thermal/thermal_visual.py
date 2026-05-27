import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
from core import Base
import argparse
import ast

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
parser.add_argument('--output_path', type=str, default='bupt/debug/vcm2bupt_m2_ablation',
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
parser.add_argument('--hm_thresh', type=float, default=0.75)
parser.add_argument('--ht_thresh', type=float, default=0.75) #阈值
parser.add_argument('--topk', type=int, default=30)



config = parser.parse_args()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义 reshape_transform 以适配 ViT 的激活
def vit_reshape_transform(tensor, height, width):
    """
    将 ViT 的激活 [batch_size, sequence_length, feature_dim] 转换为 [batch_size, channels, height, width]
    """
    batch_size, seq_len, feature_dim = tensor.shape
    # 移除 cls 和 prompt tokens，只保留 patch tokens
    num_patches = height * width  # h_resolution * w_resolution
    tensor = tensor[:, 1:num_patches + 1, :]  # 移除 cls token
    # 重塑为 [batch_size, height, width, feature_dim]
    tensor = tensor.reshape(batch_size, height, width, feature_dim)
    # 转换为 [batch_size, feature_dim, height, width]
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor

# 自定义模型包装器以提取 x12_cue
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 调用模型的 encode_image 方法
        x11, x12, xproj, x12_cue, xproj_cue = self.model.encode_image(x)
        # 返回 x12_cue 作为激活
        return x12_cue

# 数据路径
base_path = "/data1/wangzj/experience/thermal"
ir_path = os.path.join(base_path, "ir")
rgb_path = os.path.join(base_path, "rgb")
output_dir = os.path.join(base_path, "cam_outputs")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 支持的图像格式
image_extensions = (".jpg", ".jpeg", ".png")

# 处理单模态图像的函数
def process_images(image_dir, modality, cam, img_h=224, img_w=224):
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(image_extensions):
            img_path = os.path.join(image_dir, img_name)
            
            # 读取图像
            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]  # 转换为 RGB
            rgb_img = np.float32(rgb_img) / 255  # 归一化
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
            
            # 生成 CAM（无需 targets，因为 EigenCAM 不需要类别）
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"{modality}_{img_name}")
            cv2.imwrite(output_path, visualization)
            print(f"Saved CAM for {img_path} to {output_path}")

# 批量生成 CAM
def generate_cam_for_modalities():
    num_classes = config.pid_num  # 替换为你的实际类别数（如果模型中用到）
    img_h, img_w = config.img_h,config.img_w  # 替换为你的实际图像尺寸
    Model = Base(config,'1')
    Model.set_eval()

    # 包装模型以提取 x12_cue
    wrapped_model = ModelWrapper(Model.model)

    # 计算 h_resolution 和 w_resolution
    h_resolution = int((img_h - 16) // 16 + 1)  # 例如，(224-16)//16+1 = 14
    w_resolution = int((img_w - 16) // 16 + 1)

    # 初始化 EigenCAM（无监督方法）
    target_layers = [Model.model.image_encoder.ln_post]
    cam = EigenCAM(
        model=wrapped_model,
        target_layers=target_layers,
        reshape_transform=lambda x: vit_reshape_transform(x, h_resolution, w_resolution)
    )

    # 处理 ir 模态
    if os.path.exists(ir_path):
        print("Processing IR images...")
        process_images(ir_path, "ir", cam, img_h, img_w)
    
    # 处理 rgb 模态
    if os.path.exists(rgb_path):
        print("Processing RGB images...")
        process_images(rgb_path, "rgb", cam, img_h, img_w)

# 运行
if __name__ == "__main__":
    generate_cam_for_modalities()