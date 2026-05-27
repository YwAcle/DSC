import torchvision
import torch.nn as nn
from .gem_pool import GeneralizedMeanPoolingP,GeneralizedMeanPooling
import torch
import copy
import numpy as np

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

#
# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.clip_model = clip_model
#
#     def forward(self, tokenized_description):
#         # tokenized_description = clip.tokenize(description).cuda()
#         text_features = self.clip_model.encode_text(tokenized_description)
#         return text_features
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


# class PromptLearner(nn.Module):
#     def __init__(self, num_class, dtype, token_embedding):
#         super().__init__()
#
#         ctx_init = "A video of a X X X X person."
#
#         ctx_dim = 512
#         # use given words to initialize context vectors
#         ctx_init = ctx_init.replace("_", " ")
#         n_ctx = 4
#
#         tokenized_prompts = clip.tokenize(ctx_init).cuda()
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#
#         n_cls_ctx = 4
#         cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
#         nn.init.normal_(cls_vectors, std=0.02)
#         self.cls_ctx = nn.Parameter(cls_vectors)
#
#
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
#         self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])
#         self.num_class = num_class
#         self.n_cls_ctx = n_cls_ctx
#
#     def forward(self,):
#         cls_ctx = self.cls_ctx
#         # b = label.shape[0]
#         prefix = self.token_prefix.expand(self.num_class, -1, -1)
#         suffix = self.token_suffix.expand(self.num_class, -1, -1)
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (n_cls, 1, dim)
#                 cls_ctx,     # (n_cls, n_ctx, dim)
#                 suffix,  # (n_cls, *, dim)
#             ],
#             dim=1,
#         )
#
#         return prompts

class PromptLearner(nn.Module):
    def __init__(self, num_class, dtype, token_embedding,prompt_index):
        super().__init__()

        self.templates = [
            "A photo of a X X X X person.",
            "A X X X X person.",
            "A X X X X person observed in RGB and infrared sequences.",
            # "A X X X X person observed in both day and night conditions.",
            # "X X X X person."
        ]
        self.indexs = [6,1,1,1,6,6,0]
        ctx_init = self.templates[prompt_index]
        n_ctx = self.indexs[prompt_index]

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self,):
        cls_ctx = self.cls_ctx
        # b = label.shape[0]
        prefix = self.token_prefix.expand(self.num_class, -1, -1)
        suffix = self.token_suffix.expand(self.num_class, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

# class PartLearner(nn.Module):
#     def __init__(self,ctx_init, dtype, token_embedding):
#         super().__init__()
#
#         ctx_dim = 512
#         tokenized_prompts = clip.tokenize(ctx_init).cuda()
#         with torch.no_grad():
#             self.person_embedding = token_embedding(tokenized_prompts).type(dtype)
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#
#         n_cls_ctx = 4
#         cls_vectors = torch.empty(1, n_cls_ctx, ctx_dim, dtype=dtype)
#         nn.init.normal_(cls_vectors, std=0.02)
#         self.person_ctx = nn.Parameter(cls_vectors)
#
#         self.num_class = 1
#         self.n_cls_ctx = n_cls_ctx
#
#     def forward(self,):
#         prompts = torch.cat(
#             [
#                 self.person_ctx,     # (n_cls, n_ctx, dim)
#                 self.person_embedding,  # (n_cls, *, dim)
#             ],
#             dim=1,
#         )
#
#         return prompts



class Model(nn.Module):
    def __init__(self,config, num_classes, img_h, img_w):
        super(Model, self).__init__()
        self.config = config
        self.in_planes = 2048
        self.num_classes = num_classes

        self.h_resolution = int((img_h - 16) // 16 + 1)
        self.w_resolution = int((img_w - 16) // 16 + 1)
        self.vision_stride_size = 16
        clip_model,model_dict = load_clip_to_cpu('ViT-B-16', self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        self.classifier = Classifier(self.num_classes)
        self.classifier2 = Classifier2(self.num_classes)

        self.classifier_cue = Classifier(self.num_classes)
        self.classifier2_cue = Classifier2(self.num_classes)

        self.text_encoder = TextEncoder(clip_model)
        self.Gem = GeneralizedMeanPooling(norm=3.0)
        self.prompt_learner = PromptLearner(num_classes, clip_model.dtype, clip_model.token_embedding,prompt_index=config.prompt_index)
        # self.head_learner = PartLearner('head',clip_model.dtype, clip_model.token_embedding)
        # self.head_learner = PartLearner('head',clip_model.dtype, clip_model.token_embedding)
        # self.head_learner = PartLearner('head',clip_model.dtype, clip_model.token_embedding)
        # self.head_learner = PartLearner('head',clip_model.dtype, clip_model.token_embedding)
        # self.head_learner = PartLearner('head',clip_model.dtype, clip_model.token_embedding)
        # self.head_learner = PartLearner('head',clip_model.dtype, clip_model.token_embedding)

        # self.person_learner = PersonLearner(clip_model.dtype, clip_model.token_embedding)
        self.use_text_prompt_learning = True
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.seq_lenth = config.sequence_length
        self.np = self.seq_lenth
        self.nf = self.seq_lenth
        self.prompt_embedding = nn.Parameter(torch.randn(self.np, self.np, 768))
        self.prompt_cue = config.prompt_cue
        self.cls_cue = config.cls_cue
        self.STH_start_layer = config.STH_start_layer
        self.head_num = 12
        self.attn = nn.MultiheadAttention(768, self.head_num)
        self.ln_post2 = copy.deepcopy(self.image_encoder.ln_post)

    def encode_image(self, x: torch.Tensor, cv_emb=None):
        x = self.image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if cv_emb != None:
            x[:, 0] = x[:, 0] + cv_emb
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)

        # if self.prompt_cue:
        #     x = self.Prompt_interaction(x,is_start=True)

        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(11):
            if i==self.STH_start_layer:
                x = x.permute(1, 0, 2)
                x = self.Prompt_interaction(x,is_start=True)
                x = x.permute(1, 0, 2)
            elif i>self.STH_start_layer:
                x = self.Prompt_interaction(x)
            x = self.image_encoder.transformer.resblocks[i](x)

        x11 = x
        if self.prompt_cue:
            x11 = self.Prompt_interaction(x11)
        # x11 = self.image_encoder.transformer.resblocks[:11](x)
        x12 = self.image_encoder.transformer.resblocks[11](x11)

        x11 = x11.permute(1, 0, 2)  # LND -> NLD
        x12 = x12.permute(1, 0, 2)  # LND -> NLD

        # x12 = self.image_encoder.ln_post(x12)
        if self.cls_cue:
            x12_cue = self.cls_cue_interaction(x12)

        x12 = self.image_encoder.ln_post(x12)
        if self.image_encoder.proj is not None:
            xproj = x12 @ self.image_encoder.proj
            xproj_cue = x12_cue @ self.image_encoder.proj


        return x11, x12, xproj,x12_cue,xproj_cue

    def cls_cue_interaction(self,x12):
        pooled_output = x12[:, 0, :]

        bs = pooled_output.size(0) // self.nf
        D = pooled_output.size(-1)
        q = pooled_output.reshape(bs, self.nf, D).transpose(0, 1)
        kv = x12[:, -self.np:, :].reshape(bs, self.nf * self.np, D).transpose(0, 1)
        attn_output = self.attn(q, kv, kv)[0]
        pooled_output = attn_output.transpose(0, 1).reshape(bs * self.nf, D)
        pooled_output = self.image_encoder.ln_post(pooled_output)
        pooled_output = pooled_output.unsqueeze(1)
        return pooled_output

    def Prompt_interaction(self,input,is_start=False):
        if is_start:
            # prompt_embedding: num_prompts, num_prompts, embed_dim
            prompts = self.prompt_embedding
            bs = input.size(0) // self.nf
            num_chunks = self.nf // self.np
            # bs * num_frames, num_prompts, embed_dim
            prompts = prompts[None, None].repeat(bs, num_chunks, 1, 1, 1).flatten(0, 2)
            # bs * num_frames, seq_len + num_prompts, embed_dim
            x = torch.cat([input, prompts], dim=1)
            return x
        else:
            bs = input.size(1) // self.nf
            Ls = input.size(0) - self.np
            num_chunks = self.nf // self.np
            input = input.transpose(0, 1)
            hidden_states = input.reshape(bs, num_chunks, self.np, Ls + self.np, -1)
            for i in range(num_chunks):
                prompts = hidden_states[:, [i], :, Ls:, :].clone()
                hidden_states[:, [i], :, Ls:, :] = prompts.transpose(2, 3)
            hidden_states = hidden_states.flatten(0, 2)
            hidden_states = hidden_states.permute(1, 0, 2)
            return hidden_states

    def get_prompt(self,input):
        bs = input.size(0) // self.nf
        Ls = input.size(1) - self.np
        num_chunks = self.nf // self.np
        # input = input.transpose(0, 1)
        hidden_states = input.reshape(bs, num_chunks, self.np, Ls + self.np, -1)
        # for i in range(num_chunks):
        prompts = hidden_states[:, [0], :, Ls:, :]
        prompts = prompts.transpose(2,3).squeeze()
        prompts = prompts.mean(1)

        return prompts


    def forward(self, x1=None, x2=None):
        if x1 is not None and x2 is not None:
            image_features_maps = torch.cat([x1, x2], dim=0)

            image_features_last, image_features, image_features_proj,image_features_cue, image_features_proj_cue = self.encode_image(image_features_maps)

            prompt = self.get_prompt(image_features)

            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]
            image_features_cue, image_features_proj_cue = image_features_cue[:,0], image_features_proj_cue[:,0]

            # image_features = self.Gem(image_features_maps)
            img_feature_last = self.tem_pool(img_feature_last)
            img_feature = self.tem_pool(img_feature)
            img_feature_proj = self.tem_pool(img_feature_proj)

            image_features_cue = self.tem_pool(image_features_cue)
            image_features_proj_cue = self.tem_pool(image_features_proj_cue)

            cls_scores, _ = self.classifier(img_feature)
            cls_scores_proj, _ = self.classifier2(img_feature_proj)

            cls_scores_cue, _ = self.classifier_cue(image_features_cue)
            cls_scores_proj_cue, _ = self.classifier2_cue(image_features_proj_cue)


            if self.use_text_prompt_learning:
                # text side
                prompts = self.prompt_learner()
                tokenized_prompts = self.prompt_learner.tokenized_prompts
                text_features = self.text_encoder(prompts, tokenized_prompts)
                # vision side
                # video_features = img_feature_proj / img_feature_proj.norm(dim=-1, keepdim=True)
                # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # logit_scale = self.logit_scale.exp()
                logits = img_feature_proj @ text_features.t()
                logits_cue = image_features_proj_cue @text_features.t()

            return [img_feature_last,img_feature,img_feature_proj,image_features_cue,image_features_proj_cue], [cls_scores, cls_scores_proj,logits,cls_scores_cue,cls_scores_proj_cue,logits_cue],prompt

        else:
            image_features_last1, image_features1, image_features_proj1,image_features_cue1, image_features_proj_cue1 = self.encode_image(x1)
            image_features_last1 = image_features_last1[:,0]
            image_features1 = image_features1[:,0]
            image_features_proj1 = image_features_proj1[:,0]

            image_features_cue1 = image_features_cue1[:,0]
            # image_features = self.Gem(image_features_maps)
            _ = self.tem_pool(image_features_last1)
            image_features1 = self.tem_pool(image_features1)
            image_features_proj1 = self.tem_pool(image_features_proj1)

            image_features_cue1 = self.tem_pool(image_features_cue1)



            cls_scores, _ = self.classifier(image_features1)
            cls_scores_proj, _ = self.classifier2(image_features_proj1)

            return torch.cat([image_features1,image_features_cue1],dim=-1)

        # elif x1 is None and x2 is not None:
        #     image_features_last2, image_features2, image_features_proj2,image_features_cue2, image_features_proj_cue2 = self.encode_image(x2)
        #     image_features_last2 = image_features_last2[:,0]
        #     image_features2 = image_features2[:,0]
        #     image_features_proj2 = image_features_proj2[:,0]
        #
        #     image_features_cue2 = image_features_cue2[:,0]
        #
        #     # image_features = self.Gem(image_features_maps)
        #     _ = self.tem_pool(image_features_last2)
        #     image_features2 = self.tem_pool(image_features2)
        #     image_features_proj2 = self.tem_pool(image_features_proj2)
        #
        #     image_features_cue2 = self.tem_pool(image_features_cue2)
        #
        #     cls_scores, _ = self.classifier(image_features2)
        #     cls_scores_proj, _ = self.classifier2(image_features_proj2)
        #     return torch.cat([image_features2,image_features_cue2],dim=-1)

    def tem_pool(self,features,t=6):
        features = features.squeeze()
        features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
        features = features.mean(0)
        return features

    def tem_reshape(self,features,t=6):
        features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
        features = features.mean(0)
        return features



# class Model2(nn.Module):
#     def __init__(self, num_classes, img_h, img_w):
#         super(Model2, self).__init__()
#         self.in_planes = 2048
#         self.num_classes = num_classes
#
#         self.h_resolution = int((img_h - 16) // 16 + 1)
#         self.w_resolution = int((img_w - 16) // 16 + 1)
#         self.vision_stride_size = 16
#         clip_model,model_dict = load_clip_to_cpu('RN50', self.h_resolution, self.w_resolution, self.vision_stride_size)
#         clip_model.to("cuda")
#         self.image_encoder1 = nn.Sequential(clip_model.visual.conv1, clip_model.visual.bn1, clip_model.visual.conv2,
#                                             clip_model.visual.bn2, clip_model.visual.conv3, clip_model.visual.bn3,
#                                             clip_model.visual.relu, clip_model.visual.avgpool)
#         self.image_encoder2 = copy.deepcopy(self.image_encoder1)
#
#         self.image_encoder = nn.Sequential(clip_model.visual.layer1, clip_model.visual.layer2, clip_model.visual.layer3,
#                                            clip_model.visual.layer4)
#         self.attnpool = clip_model.visual.attnpool
#         self.classifier = Classifier(self.num_classes)
#         self.classifier2 = Classifier2(self.num_classes)
#
#         self.text_encoder = TextEncoder(clip_model)
#         self.Gem = GeneralizedMeanPooling(norm=3.0)
#
#
#     def forward(self, x1=None, x2=None):
#         if x1 is not None and x2 is not None:
#             image_features_map1 = self.image_encoder1(x1)
#             image_features_map2 = self.image_encoder2(x2)
#             image_features_maps = torch.cat([image_features_map1, image_features_map2], dim=0)
#
#             image_features_maps = self.image_encoder(image_features_maps)
#             image_features = self.Gem(image_features_maps)
#             image_features = self.tem_pool(image_features)
#
#             image_features_proj = self.attnpool(image_features_maps)[0]
#             image_features_proj = self.tem_pool(image_features_proj)
#
#             features, cls_scores, _ = self.classifier(image_features)
#             cls_scores_proj, _ = self.classifier2(image_features_proj)
#
#             return [features, image_features_proj], [cls_scores, cls_scores_proj]
#
#         else:
#             image_features_map1 = self.image_encoder1(x1)
#             image_features_map1 = self.image_encoder(image_features_map1)
#             image_features1 = self.Gem(image_features_map1)
#             image_features1 = self.tem_pool(image_features1)
#
#             image_features1_proj = self.attnpool(image_features_map1)[0]
#             image_features1_proj = self.tem_pool(image_features1_proj)
#
#             _, _, test_features1 = self.classifier(image_features1)
#             _, test_features1_proj = self.classifier2(image_features1_proj)
#             return torch.cat([test_features1, test_features1_proj], dim=1)
#
#
#
#     def tem_pool(self,features):
#         t = self.seq_lenth
#         features = features.squeeze()
#         features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
#         features = features.mean(0)
#         return features
#
#     # def tem_reshape(self,features,t=6):
#     #     features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
#     #     features = features.mean(0)
#     #     return features

# class RGB_Model(nn.Module):
#     def __init__(self):
#         super(RGB_Model, self,).__init__()
#         resnet = torchvision.models.resnet50(pretrained=True)
#
#         self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)
#
#     def forward(self, rgb):
#         rgb_features_map = self.resnet_conv(rgb)
#         return rgb_features_map
#
# class IR_Model(nn.Module):
#     def __init__(self):
#         super(IR_Model, self,).__init__()
#         resnet = torchvision.models.resnet50(pretrained=True)
#
#         self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)
#
#     def forward(self, ir):
#         ir_features_map = self.resnet_conv(ir)
#         return ir_features_map
#
# class Shared_Model(nn.Module):
#
#     def __init__(self):
#         super(Shared_Model, self,).__init__()
#         resnet = torchvision.models.resnet50(pretrained=True)
#         resnet.layer4[0].conv2.stride = (1, 1)
#         resnet.layer4[0].downsample[0].stride = (1, 1)
#
#         self.resnet_conv = nn.Sequential(resnet.layer1,
#                                          resnet.layer2, resnet.layer3, resnet.layer4)
#         self.GAP = GeneralizedMeanPoolingP()
#
#     def forward(self, x):
#         features_map = self.resnet_conv(x)
#         features = self.GAP(features_map)
#         return features_map,features

class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        self.GEM = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(768)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(768, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        # features = self.GEM(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return cls_score, self.l2_norm(bn_features)

class Classifier2(nn.Module):
    def __init__(self, pid_num):
        super(Classifier2, self, ).__init__()
        self.pid_num = pid_num
        self.BN = nn.BatchNorm1d(512)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(512, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return cls_score, self.l2_norm(features)


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model,model.state_dict()