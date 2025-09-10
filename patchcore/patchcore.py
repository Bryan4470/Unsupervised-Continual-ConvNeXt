"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

from timm.models import create_model
import argparse
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn.cluster import KMeans

from patchcore.convnext_dualprompt import debug_print_dual_prompt
from patchcore.convnext_dualprompt import ConvNeXtDualPrompt

import copy

LOGGER = logging.getLogger(__name__)

class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        key_fps_k,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        self.dataloader_count = 0

        self.key_fps_k=key_fps_k

        convnext_variant = kwargs.get("convnext_variant", "convnext_base_in22ft1k")
        injection_spec   = kwargs.get("prompt_inject", "all")   # inject many by default (for ablations)
        analysis_site    = kwargs.get("analysis_site", "s3b6")  # single fixed analysis layer

        print("injection_spec", injection_spec)

        # -----------------------------
        # FROZEN extractor (no prompt training) for keys/prototypes
        # -----------------------------
        self.model = ConvNeXtDualPrompt(
            variant=convnext_variant,
            pretrained=True,
            proj_dim=1024,  # 768 tiny/small, 1024 base, 1536 large
            injection_spec=injection_spec,
            analysis_site=analysis_site,
            base_input_hw=self.input_shape[-2:],
            key_fps_k=key_fps_k
        ).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # -----------------------------
        # Trainable prompt model (same analysis site)
        # -----------------------------
        self.prompt_model = ConvNeXtDualPrompt(
            variant=convnext_variant,
            pretrained=True,
            proj_dim=1024,
            injection_spec=injection_spec,
            analysis_site=analysis_site,
            base_input_hw=self.input_shape[-2:],
            key_fps_k=key_fps_k
        ).to(self.device)

        # Freeze backbone in prompt_model; only prompts + heads learn
        for n, p in self.prompt_model.named_parameters():
            if n.startswith("model."):
                p.requires_grad = False
        for n, p in self.prompt_model.named_parameters():
            if ("kernel_prompt" in n) or ("mask_prompt" in n) or n.startswith(("seg_proj","cls_proj")):
                p.requires_grad = True

        self.prompt_model.train_contrastive = True

        # Debug print
        debug_print_dual_prompt(self.prompt_model)

        trainable = [p for n,p in self.prompt_model.named_parameters() if p.requires_grad]
        frozen    = [p for n,p in self.prompt_model.named_parameters() if not p.requires_grad]
        print(f"Trainable params: {sum(p.numel() for p in trainable)}")
        print(f"Frozen params:    {sum(p.numel() for p in frozen)}")

        froz_trainable = [p for n,p in self.model.named_parameters() if p.requires_grad]
        froz_frozen    = [p for n,p in self.model.named_parameters() if not p.requires_grad]
        print(f"Trainable params Frozen: {sum(p.numel() for p in froz_trainable)}")
        print(f"Frozen params Frozen:    {sum(p.numel() for p in froz_frozen)}")

        # Rebuild preprocessing / aggregator to match ConvNeXt token dim from analysis stage
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.input_shape[-2:], device=self.device)
            token_dim = self.model(dummy, train=False)["seg_feat"][0].shape[-1]
        self.forward_modules["preprocessing"] = patchcore.common.Preprocessing(
            [token_dim], pretrain_embed_dimension
        )
        self.forward_modules["preadapt_aggregator"] = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        ).to(self.device)

    def set_dataloadercount(self, dataloader_count):
        self.dataloader_count = dataloader_count

    # ------------------- embedding (frozen) -------------------
    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""
        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            out = self.model(images, train=False)
            tokens_list = out["seg_feat"]  # list of (B, N, D); here length == 1
            featmaps = [self._tokens_to_feature_maps(t, images) for t in tokens_list]

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in featmaps]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]

        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:])
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    # ------------------- embedding (prompted) -------------------
    def embed_prompt(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed_prompt(input_image))
            return features
        return self._embed_prompt(data)

    def _embed_prompt(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images (using prompt model)."""
        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            out = self.prompt_model(images, train=False)
            tokens_list = out["seg_feat"]  # list of (B, N, D)
            featmaps = [self._tokens_to_feature_maps(t, images) for t in tokens_list]

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in featmaps]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]

        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:])
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def _tokens_to_feature_maps(self, tokens: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Convert ConvNeXt tokens (B, N, D) to feature maps (B, D, Ht, Wt).
        We infer (Ht, Wt) from the input stride (analysis stage ~ /16 typically) and fall back to sqrt(N).
        """
        B, N, D = tokens.shape
        ht = max(1, images.shape[-2] // 16)
        wt = max(1, images.shape[-1] // 16)
        if ht * wt != N:
            s = int(round(N ** 0.5))
            if s * s == N:
                ht = wt = s
            else:
                ht = max(1, int(np.floor(N ** 0.5)))
                wt = max(1, N // ht)
        fm = tokens.view(B, ht, wt, D).permute(0, 3, 1, 2).contiguous()
        return fm

    # ------------------- training helpers -------------------
    def _embed_train(self, images, detach=True, provide_patch_shapes=False):
        return self.prompt_model(images, task_id=self.dataloader_count, train=True)

    def _embed_train_false(self, images, detach=True, provide_patch_shapes=False):
        return self.prompt_model(images, task_id=self.dataloader_count, train=False)

    def _embed_train_sam(self, images, detach=True, provide_patch_shapes=False, image_path=None):
        return self.prompt_model(images, task_id=self.dataloader_count, train=True, image_path=image_path)

    # ------------------- memory fitting -------------------
    def fit(self, training_data):
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])

    def fit_with_return_feature(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(training_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])
        return features

    def get_all_features(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(training_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))
        features = np.concatenate(features, axis=0)
        return features

    def fit_with_limit_size(self, training_data, limit_size):
        return self._fill_memory_bank_with_limit_size(training_data, limit_size)

    def _fill_memory_bank_with_limit_size(self, input_data, limit_size):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        print(features.shape)
        return features

    def get_mem_limit_size(self, training_data, limit_size):
        return self._get_mem_limit_size(training_data, limit_size)

    def _get_mem_limit_size(self, input_data, limit_size):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        return features

    def fit_with_limit_size_prompt(self, training_data, limit_size):
        return self._fill_memory_bank_with_limit_size_prompt(training_data, limit_size)

    def _fill_memory_bank_with_limit_size_prompt(self, input_data, limit_size):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed_prompt(input_image)

        features = []
        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    # ------------------- prototypes / keys -------------------
    def get_normal_prototypes(self, data, args):
        with torch.no_grad():
            cls_memory = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                        output = self.model(image)
                        cls_features = output["pre_logits"]  # (B, D)
                        cls_memory.append(cls_features.cpu())
        cls_prototypes = torch.cat([cls_f for cls_f in cls_memory], dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size, random_state=0)
        labels = kmeans.fit_predict(cls_prototypes)
        representatives = torch.zeros(args.prototype_size, cls_prototypes.shape[1])
        for i in range(args.prototype_size):
            cluster_tensors = cls_prototypes[labels == i]
            representative = np.mean(cluster_tensors, axis=0)
            representatives[i] = torch.from_numpy(representative)
        return representatives

    def get_normal_prototypes_instance(self, data, args):
        with torch.no_grad():
            cls_memory = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                        output = self.model(image)
                        cls_features = output["pre_logits"]
                        cls_memory.append(cls_features.cpu())
        cls_prototypes = torch.cat([cls_f for cls_f in cls_memory], dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size, n_init=10, max_iter=300).fit(cls_prototypes)
        centers = kmeans.cluster_centers_
        representatives = torch.from_numpy(centers).float()
        return representatives

    def get_normal_prototypes_seg(self, data, args):
        with torch.no_grad():
            seg_feat_memory = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                        output = self.model(image)
                        seg_tokens = output["seg_feat"][0]  # (B, N, D)
                        seg_feat_memory.append(seg_tokens.cpu())
        flat = torch.cat([ft.reshape(ft.shape[0], -1) for ft in seg_feat_memory], dim=0).numpy()  # (M, N*D)
        kmeans = KMeans(n_clusters=args.prototype_size, n_init=10, max_iter=300).fit(flat)
        centers = kmeans.cluster_centers_  # (K, N*D)
        last = seg_feat_memory[-1]
        N, D = last.shape[1], last.shape[2]
        representatives = torch.from_numpy(centers).float().reshape(args.prototype_size, N, D)
        return representatives

    def get_normal_prototypes_seg_mean(self, data, args):
        with torch.no_grad():
            seg_feat_memory = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                        output = self.model(image)
                        seg_tokens = output["seg_feat"][0]  # (B, N, D)
                        seg_feat_memory.append(seg_tokens.cpu())
        flat = torch.cat([ft.reshape(ft.shape[0], -1) for ft in seg_feat_memory], dim=0).numpy()  # (M, N*D)
        kmeans = KMeans(n_clusters=args.prototype_size, random_state=0)
        labels = kmeans.fit_predict(flat)
        last = seg_feat_memory[-1]
        N, D = last.shape[1], last.shape[2]
        representatives = torch.zeros(args.prototype_size, N, D)
        for i in range(args.prototype_size):
            cluster_tensors = flat[labels == i]
            representative = np.mean(cluster_tensors, axis=0)
            representatives[i] = torch.from_numpy(representative).reshape(N, D)
        return representatives

    def get_task_keys(self, data):
        with torch.no_grad():
            key_memory = []
            with tqdm.tqdm(data, desc="extracting keys...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image_tensor = image["image"].to(torch.float).to(self.device)
                    else:
                        image_tensor = image.to(torch.float).to(self.device)
                    output = self.model(image_tensor, train=False)
                    pre_logits = output["pre_logits"]  # (B, D)
                    key_memory.append(pre_logits.cpu())
        keys = torch.cat(key_memory, dim=0)
        return keys

    # ------------------- optimizer -------------------
    def _build_prompt_optimizer(self,
                            lr_prompt=2e-4, wd_prompt=0.0,
                            lr_head=5e-4,   wd_head=0.05):
        prompts, heads = [], []
        for n, p in self.prompt_model.named_parameters():
            if not p.requires_grad:
                continue
            if ("kernel_prompt" in n) or ("mask_prompt" in n):
                prompts.append(p)
            elif n.startswith(("seg_proj", "cls_proj")):
                heads.append(p)
        param_groups = []
        if prompts:
            param_groups.append({"params": prompts, "lr": lr_prompt, "weight_decay": wd_prompt})
        if heads:
            param_groups.append({"params": heads,   "lr": lr_head,   "weight_decay": wd_head})
        if not param_groups:
            param_groups = [{"params": [p for p in self.prompt_model.parameters() if p.requires_grad]}]
        return torch.optim.AdamW(param_groups)

    # ------------------- training loops -------------------
    def train(self, data, dataloader_count, memory_feature):
        args = np.load('../args_dict.npy', allow_pickle=True).item()
        args.prototype_size = 5
        args.lr = 0.0005
        args.decay_epochs = 3
        args.warmup_epochs = 1
        args.cooldown_epochs = 1
        args.patience_epochs = 1
        optimizer = self._build_prompt_optimizer(lr_prompt=2e-4, wd_prompt=0.0,
                                         lr_head=5e-4,   wd_head=0.05)
        self.dataloader_count = dataloader_count

        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        else:
            lr_scheduler = None

        prompt_cls_feature = self.get_normal_prototypes(data, args=args)
        self.prompt_model.set_prompt_cls(dataloader_count, prompt_cls_feature)
        prompt_seg_feature = self.get_normal_prototypes_seg(data, args=args)
        self.prompt_model.set_prompt_seg(dataloader_count, prompt_seg_feature)

        epochs = 10
        self.prompt_model.train()
        for i in range(epochs):
            loss_list = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                    res = self._embed_train(image, provide_patch_shapes=True)
                    loss = res['loss']
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), args.clip_grad)
                    optimizer.step()
            print(f"epoch:{i} loss:{np.mean(loss_list)}")
            if lr_scheduler:
                lr_scheduler.step(i)
        return prompt_seg_feature

    def train_contrastive(self, data, dataloader_count, memory_feature=None):
        args = np.load('../args_dict.npy', allow_pickle=True).item()
        args.prototype_size = 5
        args.lr = 0.0005
        args.decay_epochs = 10
        args.warmup_epochs = 2
        args.cooldown_epochs = 3
        args.patience_epochs = 3
        optimizer = self._build_prompt_optimizer(lr_prompt=2e-4, wd_prompt=0.0,
                                         lr_head=5e-4,   wd_head=0.05)
        self.dataloader_count = dataloader_count

        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        else:
            lr_scheduler = None

        epochs = 20
        self.prompt_model.train()
        self.prompt_model.train_contrastive = True
        for i in range(epochs):
            loss_list = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if image["image"].shape[0] < 2:
                        continue
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                    res = self._embed_train_false(image, provide_patch_shapes=True)
                    loss = res['loss']
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), args.clip_grad)
                    optimizer.step()
            print(f"epoch:{i} loss:{np.mean(loss_list)}")
            if lr_scheduler:
                lr_scheduler.step(i)

    def train_sam(self, data, dataloader_count, memory_feature=None):
        args = np.load('../args_dict.npy', allow_pickle=True).item()
        args.lr = 0.0005
        args.decay_epochs = 3
        args.warmup_epochs = 1
        args.cooldown_epochs = 1
        args.patience_epochs = 1
        optimizer = self._build_prompt_optimizer(lr_prompt=2e-4, wd_prompt=0.0,
                                         lr_head=5e-4,   wd_head=0.05)
        self.dataloader_count = dataloader_count

        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        else:
            lr_scheduler = None

        epochs = 10
        self.prompt_model.train()
        self.prompt_model.train_contrastive = True
        for i in range(epochs):
            loss_list = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image_paths = image["image_path"]
                        image = image["image"].cuda()
                    res = self._embed_train_sam(image, provide_patch_shapes=True, image_path=image_paths)
                    loss = res['loss']
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    if loss != 0:
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), args.clip_grad)
                    optimizer.step()
            print(f"epoch:{i} loss:{np.mean(loss_list)}")
            if lr_scheduler:
                lr_scheduler.step(i)

    # ------------------- inference -------------------
    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        _ = self.forward_modules.eval()
        scores, masks, labels_gt, masks_gt = [], [], [], []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()
        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
        return [score for score in image_scores], [mask for mask in masks]

    def predict_prompt(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_prompt(data)
        return self._predict_prompt(data)

    def _predict_dataloader_prompt(self, dataloader):
        _ = self.forward_modules.eval()
        scores, masks, labels_gt, masks_gt = [], [], [], []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict_prompt(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict_prompt(self, images):
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()
        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed_prompt(images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
        return [score for score in image_scores], [mask for mask in masks]

    def export_feature_pipeline_state(self):
        return {
            "preprocessing": copy.deepcopy(self.forward_modules["preprocessing"].state_dict() ),
            "preadapt_aggregator": copy.deepcopy(self.forward_modules["preadapt_aggregator"].state_dict()),
        }

    def import_feature_pipeline_state(self, state_dicts):
        self.forward_modules["preprocessing"].load_state_dict(state_dicts["preprocessing"])
        self.forward_modules["preadapt_aggregator"].load_state_dict(state_dicts["preadapt_aggregator"])

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(save_path, save_features_separately=False, prepend=prepend)
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules["preprocessing"].output_dim,
            "target_embed_dimension": self.forward_modules["preadapt_aggregator"].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(patchcore_params["backbone.name"])
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)
        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
