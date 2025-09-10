import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch
import tqdm
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label
from bisect import bisect
import time
import cv2
from timm.optim import create_optimizer

from typing import List

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}

@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--memory_size", type=int, default=196, show_default=True)
@click.option("--epochs_num", type=int, default=25, show_default=True)
@click.option("--key_size", type=int, default=196, show_default=True)
@click.option("--basic_size", type=int, default=1960, show_default=True)
def main(**kwargs):
    pass

@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
    memory_size,
    epochs_num,
    key_size,
    basic_size,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )
    run_save_path_nolimit = patchcore.utils.create_storage_folder(
        results_path + '_nolimit', log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)
    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device(f"cuda:{device.index}") if "cuda" in device.type.lower() else contextlib.suppress()
    )

    result_collect = []
    result_collect_nolimit = []
    key_feature_list = [0] * len(list_of_dataloaders)
    memory_feature_list = [0] * len(list_of_dataloaders)
    prompt_list = [0] * len(list_of_dataloaders)
    pipeline_states = [None] * len(list_of_dataloaders)
    LOGGER.info(f"Total tasks (datasets) to process: {len(list_of_dataloaders)}")

    # #check if gradient is updated
    # def log_prompt_grad_norms(model):
    #     for name, p in model.named_parameters():
    #         if 'kernel_prompt' in name or 'mask_prompt' in name or 'mask_scale' in name:
    #             if p.grad is not None:
    #                 print(f'{name}: grad_norm={p.grad.detach().norm().item():.4e}')
    #                 pass
    #             else:
    #                 print(f'{name}: grad_norm=None')

    with device_context:
        torch.cuda.empty_cache()
        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            LOGGER.info(f"Evaluating dataset [{dataloaders['training'].name}] ({dataloader_count + 1}/{len(list_of_dataloaders)})...")

            patchcore.utils.fix_seeds(seed, device)
            dataset_name = dataloaders["training"].name

            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](device)
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)

            if len(PatchCore_list) > 1:
                LOGGER.info(f"Utilizing PatchCore Ensemble (N={len(PatchCore_list)}).")
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(f"Training models ({i + 1}/{len(PatchCore_list)})")
                torch.cuda.empty_cache()

                PatchCore.set_dataloadercount(dataloader_count)
                # Build base keys (KPK)
                key_feature = PatchCore.fit_with_limit_size(dataloaders["training"], key_size)
                key_feature_list[dataloader_count] = key_feature

                # ======== Keys from frozen mid-layer patches (analysis layer) ========
                PatchCore.prompt_model.eval()
                all_key_patches = []
                with torch.no_grad():
                    for batch in dataloaders["training"]:
                        if isinstance(batch, dict):
                            imgs = batch["image"].to(device).float()
                        else:
                            imgs = batch.to(device).float()
                        tokens = PatchCore.model.extract_key_patches(imgs)
                        all_key_patches.append(tokens.reshape(-1, tokens.shape[-1]).cpu())
                if len(all_key_patches) == 0:
                    raise RuntimeError("No training images encountered when building KEYS.")
                keys_tensor_full = torch.cat(all_key_patches, dim=0)
                PatchCore.prompt_model.set_task_keys(dataloader_count, keys_tensor_full)
                key_feature_list[dataloader_count] = PatchCore.prompt_model.keys_by_task[dataloader_count].detach().cpu().numpy()

            aggregator = {"scores": [], "segmentations": []}
            basic_aggregator = {"scores": [], "segmentations": []}
            start_time = time.time()
            pr_auroc = 0
            basic_pr_auroc = 0

            args = np.load('./args_dict.npy', allow_pickle=True).item()
            args.lr = 0.0005
            args.decay_epochs = 15
            args.warmup_epochs = 3
            args.cooldown_epochs = 5
            args.patience_epochs = 5
            optimizer = create_optimizer(args, PatchCore.prompt_model)
            epochs = epochs_num
            PatchCore.prompt_model.train()
            PatchCore.prompt_model.train_contrastive = True
            if args.sched != 'constant':
                lr_scheduler, _ = patchcore.scheduler.create_scheduler(args, optimizer)
            else:
                lr_scheduler = None

            best_auroc = best_full_pixel_auroc = best_img_ap = 0
            best_pixel_ap = best_pixel_pro = best_time_cost = 0

            for epoch in range(epochs):
                for i, PatchCore in enumerate(PatchCore_list):
                    torch.cuda.empty_cache()
                    PatchCore.prompt_model.train()
                    loss_list = []
                    with tqdm.tqdm(dataloaders["training"], desc="training...", leave=False) as data_iterator:
                        for image in data_iterator:
                            if isinstance(image, dict):
                                image_paths = image["image_path"]
                                image = image["image"].cuda()
                            res = PatchCore._embed_train_sam(image, provide_patch_shapes=True, image_path=image_paths)
                            loss = res['loss']
                            loss_list.append(loss.item())
                            optimizer.zero_grad()
                            if loss != 0:
                                loss.backward()

                                #check if gradient is updated
                                # log_prompt_grad_norms(PatchCore.prompt_model)
                            torch.nn.utils.clip_grad_norm_(PatchCore.prompt_model.parameters(), args.clip_grad)
                            optimizer.step()
                        print(f"epoch:{epoch} loss:{np.mean(loss_list)}")
                    if lr_scheduler:
                        lr_scheduler.step(epoch)

                    PatchCore.prompt_model.eval()
                    # No-limit memory evaluation
                    nolimit_memory_feature = PatchCore.fit_with_limit_size_prompt(dataloaders["training"], basic_size)
                    PatchCore.anomaly_scorer.fit(detection_features=[nolimit_memory_feature])
                    basic_scores, basic_segmentations, basic_labels_gt, basic_masks_gt = PatchCore.predict_prompt(dataloaders["testing"])
                    basic_aggregator["scores"].append(basic_scores)
                    basic_aggregator["segmentations"].append(basic_segmentations)
                    basic_end_time = time.time()

                    # Limited memory evaluation
                    memory_feature = PatchCore.fit_with_limit_size_prompt(dataloaders["training"], memory_size)
                    PatchCore.anomaly_scorer.fit(detection_features=[memory_feature])
                    scores, segmentations, labels_gt, masks_gt = PatchCore.predict_prompt(dataloaders["testing"])
                    aggregator["scores"].append(scores)
                    aggregator["segmentations"].append(segmentations)
                    end_time = time.time()

                # Aggregate limited-memory ensemble results
                scores_arr = np.array(aggregator["scores"])
                min_scores = scores_arr.min(axis=-1).reshape(-1, 1)
                max_scores = scores_arr.max(axis=-1).reshape(-1, 1)
                scores_arr = (scores_arr - min_scores) / (max_scores - min_scores + 1e-12)
                scores_arr = np.mean(scores_arr, axis=0)
                segmentations_arr = np.array(aggregator["segmentations"])
                min_seg = segmentations_arr.reshape(len(segmentations_arr), -1).min(axis=-1).reshape(-1, 1, 1, 1)
                max_seg = segmentations_arr.reshape(len(segmentations_arr), -1).max(axis=-1).reshape(-1, 1, 1, 1)
                segmentations_arr = (segmentations_arr - min_seg) / (max_seg - min_seg + 1e-12)
                segmentations_arr = np.mean(segmentations_arr, axis=0)

                time_cost = (end_time - basic_end_time) / len(dataloaders["testing"])
                anomaly_labels = [x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate]
                ap_seg_flat = segmentations_arr.flatten()
                auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(scores_arr, anomaly_labels)["auroc"]
                ap_mask_flat = np.array(masks_gt).flatten().astype(np.int32)
                pixel_ap = average_precision_score(ap_mask_flat, ap_seg_flat)

                # Aggregate no-limit ensemble results
                basic_scores_arr = np.array(basic_aggregator["scores"])
                bmin = basic_scores_arr.min(axis=-1).reshape(-1, 1)
                bmax = basic_scores_arr.max(axis=-1).reshape(-1, 1)
                basic_scores_arr = (basic_scores_arr - bmin) / (bmax - bmin + 1e-12)
                basic_scores_arr = np.mean(basic_scores_arr, axis=0)
                basic_segs_arr = np.array(basic_aggregator["segmentations"])
                bmin_seg = basic_segs_arr.reshape(len(basic_segs_arr), -1).min(axis=-1).reshape(-1, 1, 1, 1)
                bmax_seg = basic_segs_arr.reshape(len(basic_segs_arr), -1).max(axis=-1).reshape(-1, 1, 1, 1)
                basic_segs_arr = (basic_segs_arr - bmin_seg) / (bmax_seg - bmin_seg + 1e-12)
                basic_segs_arr = np.mean(basic_segs_arr, axis=0)
                basic_time_cost = (basic_end_time - start_time) / len(dataloaders["testing"])
                basic_anomaly_labels = [x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate]
                basic_ap_seg_flat = basic_segs_arr.flatten()
                basic_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(basic_scores_arr, basic_anomaly_labels)["auroc"]
                basic_ap_mask_flat = np.array(basic_masks_gt).flatten().astype(np.int32)
                basic_pixel_ap = average_precision_score(basic_ap_mask_flat, basic_ap_seg_flat)

                # Record best limited-memory results
                if auroc > pr_auroc:
                    memory_feature_list[dataloader_count] = memory_feature
                    prompt_list[dataloader_count] = PatchCore.prompt_model.get_cur_prompt()
                    pipeline_states[dataloader_count] = PatchCore.export_feature_pipeline_state()

                    if pr_auroc != 0:
                        result_collect.pop()
                    pr_auroc = auroc
                    img_ap = average_precision_score(anomaly_labels, scores_arr)
                    segmentations_224 = np.array([cv2.resize(seg, (224, 224)) for seg in segmentations_arr])
                    if save_segmentation_images:
                        image_paths = [x[2] for x in dataloaders["testing"].dataset.data_to_iterate]
                        mask_paths = [x[3] for x in dataloaders["testing"].dataset.data_to_iterate]

                        def image_transform(image):
                            in_std = np.array(dataloaders["testing"].dataset.transform_std).reshape(-1, 1, 1)
                            in_mean = np.array(dataloaders["testing"].dataset.transform_mean).reshape(-1, 1, 1)
                            image = dataloaders["testing"].dataset.transform_img(image)
                            return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)

                        def mask_transform(mask):
                            return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                        image_save_path = os.path.join(run_save_path, "segmentation_images", dataset_name)
                        os.makedirs(image_save_path, exist_ok=True)
                        patchcore.utils.plot_segmentation_images(
                            image_save_path,
                            image_paths,
                            segmentations_224,
                            scores_arr,
                            mask_paths,
                            image_transform=image_transform,
                            mask_transform=mask_transform,
                        )
                    pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(segmentations_224, masks_gt)
                    full_pixel_auroc = pixel_scores["auroc"]
                    sel_idxs = [idx for idx, m in enumerate(masks_gt) if np.sum(m) > 0]
                    pixel_scores_anomaly = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                        [segmentations_224[i] for i in sel_idxs],
                        [masks_gt[i] for i in sel_idxs],
                    )
                    anomaly_pixel_auroc = pixel_scores_anomaly["auroc"]
                    for idx, m in enumerate(masks_gt):
                        masks_gt[idx] = np.array(m[0])
                    for idx, s in enumerate(segmentations_224):
                        segmentations_224[idx] = np.array(s)
                    pixel_pro, _ = calculate_au_pro(np.array(masks_gt), np.array(segmentations_224))
                    result_collect.append({
                        "dataset_name": dataset_name,
                        "instance_auroc": auroc,
                        "full_pixel_auroc": full_pixel_auroc,
                        "anomaly_pixel_auroc": anomaly_pixel_auroc,
                        "image_ap": img_ap,
                        "pixel_ap": pixel_ap,
                        "pixel_pro": pixel_pro,
                        "time_cost:": time_cost
                    })
                    print(f"current task:{dataloader_count+1}/train task:{dataloader_count+1}, "
                          f"image_auc:{auroc}, pixel_auc:{full_pixel_auroc}, image_ap:{img_ap}, pixel_ap:{pixel_ap}, pixel_pro:{pixel_pro}, time_cost:{time_cost}")
                    best_auroc, best_full_pixel_auroc = auroc, full_pixel_auroc
                    best_img_ap, best_pixel_ap, best_pixel_pro, best_time_cost = img_ap, pixel_ap, pixel_pro, time_cost

                # Record best no-limit results (for CSV output)
                if basic_auroc > basic_pr_auroc:
                    if basic_pr_auroc != 0:
                        result_collect_nolimit.pop()
                    basic_pr_auroc = basic_auroc
                    basic_img_ap = average_precision_score(basic_anomaly_labels, basic_scores_arr)
                    basic_seg_224 = np.array([cv2.resize(seg, (224, 224)) for seg in basic_segs_arr])
                    basic_pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(basic_seg_224, basic_masks_gt)
                    basic_full_pixel_auroc = basic_pixel_scores["auroc"]
                    basic_sel_idxs = [idx for idx, m in enumerate(basic_masks_gt) if np.sum(m) > 0]
                    basic_pixel_scores_anom = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                        [basic_seg_224[i] for i in basic_sel_idxs],
                        [basic_masks_gt[i] for i in basic_sel_idxs],
                    )
                    basic_anomaly_pixel_auroc = basic_pixel_scores_anom["auroc"]
                    for idx, m in enumerate(basic_masks_gt):
                        basic_masks_gt[idx] = np.array(m[0])
                    for idx, s in enumerate(basic_seg_224):
                        basic_seg_224[idx] = np.array(s)
                    basic_pixel_pro, _ = calculate_au_pro(np.array(basic_masks_gt), np.array(basic_seg_224))
                    result_collect_nolimit.append({
                        "dataset_name": dataset_name,
                        "instance_auroc": basic_auroc,
                        "full_pixel_auroc": basic_full_pixel_auroc,
                        "anomaly_pixel_auroc": basic_anomaly_pixel_auroc,
                        "image_ap": basic_img_ap,
                        "pixel_ap": basic_pixel_ap,
                        "pixel_pro": basic_pixel_pro,
                        "time_cost:": basic_time_cost
                    })

            print(f"Limited current task:{dataloader_count+1}/train task:{dataloader_count+1}, "
                  f"image_auc:{best_auroc}, pixel_auc:{best_full_pixel_auroc}, image_ap:{best_img_ap}, "
                  f"pixel_ap:{best_pixel_ap}, pixel_pro:{best_pixel_pro}, time_cost:{best_time_cost}")
            print(f"Nolimited current task:{dataloader_count+1}/train task:{dataloader_count+1}, "
                  f"image_auc:{basic_auroc}, pixel_auc:{basic_full_pixel_auroc}, image_ap:{basic_img_ap}, "
                  f"pixel_ap:{basic_pixel_ap}, pixel_pro:{basic_pixel_pro}, time_cost:{basic_time_cost}")

            if save_patchcore_model:
                patchcore_save_path = os.path.join(run_save_path, "models", dataset_name)
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (f"Ensemble-{i+1}-{len(PatchCore_list)}_" if len(PatchCore_list) > 1 else "")
                    PatchCore.save_to_path(patchcore_save_path, prepend)

            LOGGER.info("\n\n-----\n")

    # --------- Unified task-agnostic inference (per-dataloader selection) ----------
    LOGGER.info("Running unified task-agnostic inference on all tasks...")
    result_collect_final = []

    try:
        ucad_key_feature_list
    except NameError:
        ucad_key_feature_list = [None] * len(list_of_dataloaders)

    imagesize0 = list_of_dataloaders[0]["training"].dataset.imagesize
    sampler0 = methods["get_sampler"](device)
    PC_infer = methods["get_patchcore"](imagesize0, sampler0, device)[0]
    PC_infer.prompt_model.eval()

    for tid in range(len(list_of_dataloaders)):
        keys_np = key_feature_list[tid]
        keys_tensor = (
            torch.from_numpy(keys_np).to(device).float()
            if isinstance(keys_np, np.ndarray)
            else keys_np.to(device).float()
        )
        PC_infer.prompt_model.set_task_keys(tid, keys_tensor)

    def build_calibration_batch(dl, device, limit=32):
        imgs = []
        with torch.no_grad():
            for batch in dl:
                x = batch["image"] if isinstance(batch, dict) else batch
                x = x.to(device).float()
                imgs.append(x)
                if sum(t.shape[0] for t in imgs) >= limit:
                    break
        if not imgs:
            raise RuntimeError("Empty test dataloader.")
        X = torch.cat(imgs, dim=0)
        return X[:limit]

    @torch.no_grad()
    def kpk_topk_for_image(img1: torch.Tensor, K: int = 3) -> List[int]:
        import torch.nn.functional as F
        PC_infer.prompt_model.eval()
        q = PC_infer.model.extract_key_patches(img1)[0]
        q = F.normalize(q, dim=-1)
        scores = []
        for tid, keys in PC_infer.prompt_model.keys_by_task.items():
            ks = F.normalize(keys.to(q.device), dim=-1)
            d = 1.0 - torch.matmul(q, ks.T)
            scores.append((tid, float(d.min(dim=1).values.mean())))
        scores.sort(key=lambda t: t[1])
        return [tid for tid, _ in scores[:K]]

    @torch.no_grad()
    def ensure_ucad_keys_for_task(tid: int):
        if ucad_key_feature_list[tid] is not None and not isinstance(ucad_key_feature_list[tid], int):
            return
        if pipeline_states[tid] is not None:
            PC_infer.import_feature_pipeline_state(pipeline_states[tid])
        else:
            _ = PC_infer.fit_with_limit_size_prompt(list_of_dataloaders[tid]["training"], memory_size)
        ucad_keys = PC_infer.fit_with_limit_size(list_of_dataloaders[tid]["training"], key_size)
        ucad_key_feature_list[tid] = ucad_keys

    @torch.no_grad()
    def ucad_mean_score_for_task(tid: int, imgs: torch.Tensor) -> float:
        ensure_ucad_keys_for_task(tid)
        if pipeline_states[tid] is not None:
            PC_infer.import_feature_pipeline_state(pipeline_states[tid])
        PC_infer.anomaly_scorer.fit(detection_features=[ucad_key_feature_list[tid]])
        s, _ = PC_infer._predict(imgs)
        return float(np.mean(s))

    for tid_loop, dataloaders in enumerate(list_of_dataloaders):
        scores_task, masks_task, labels_gt_task, masks_gt_task = [], [], [], []
        cal_imgs = build_calibration_batch(dataloaders["testing"], device, limit=32)

        vote = {}
        for b in range(cal_imgs.shape[0]):
            cand = kpk_topk_for_image(cal_imgs[b:b+1], K=min(3, len(list_of_dataloaders)))
            for t in cand:
                vote[t] = vote.get(t, 0) + 1
        cand_sorted = sorted(vote.items(), key=lambda kv: kv[1], reverse=True)
        candidates = [t for t, _ in cand_sorted[:min(3, len(cand_sorted))]] or list(range(len(list_of_dataloaders)))

        ucad_scores = [(t, ucad_mean_score_for_task(t, cal_imgs)) for t in candidates]
        best_tid = min(ucad_scores, key=lambda kv: kv[1])[0]

        if pipeline_states[best_tid] is not None:
            PC_infer.import_feature_pipeline_state(pipeline_states[best_tid])
        else:
            _ = PC_infer.fit_with_limit_size_prompt(list_of_dataloaders[best_tid]["training"], memory_size)

        PC_infer.prompt_model.set_prompt_task(best_tid, prompt_list[best_tid])
        PC_infer.prompt_model.eval()

        mem = np.asarray(memory_feature_list[best_tid], dtype=np.float32)
        PC_infer.anomaly_scorer.fit(detection_features=[mem])

        total_infer_seconds = 0.0
        num_proc_images = 0

        with tqdm.tqdm(
            dataloaders["testing"],
            desc=f"Unified (task={best_tid}) {tid_loop+1}/{len(list_of_dataloaders)}",
            leave=False,
        ) as data_iterator:
            for sample in data_iterator:
                if isinstance(sample, dict):
                    labels_gt_task.extend(
                        sample["is_anomaly"].detach().cpu().numpy().astype(int).reshape(-1).tolist()
                    )
                    masks = sample["mask"]
                    if torch.is_tensor(masks):
                        m = masks.detach().cpu()
                        if m.ndim == 4 and m.shape[1] == 1:
                            m = m[:, 0]
                        elif m.ndim != 3:
                            raise ValueError(f"Unexpected mask shape: {m.shape}")
                        masks_gt_task.extend([mm.numpy() for mm in m])
                    else:
                        m = np.array(masks)
                        if m.ndim == 4 and m.shape[1] == 1:
                            m = m[:, 0]
                        elif m.ndim != 3:
                            raise ValueError(f"Unexpected mask shape (np): {m.shape}")
                        masks_gt_task.extend([m[i] for i in range(m.shape[0])])

                    images = sample["image"].to(device).float()
                    bsz = images.shape[0]
                else:
                    bsz = sample.shape[0] if torch.is_tensor(sample) and sample.ndim == 4 else 1
                    labels_gt_task.extend([0] * bsz)
                    masks_gt_task.extend([
                        np.zeros(
                            (dataloaders["training"].dataset.imagesize, dataloaders["training"].dataset.imagesize),
                            dtype=np.uint8
                        ) for _ in range(bsz)
                    ])
                    images = sample.to(device).float() if torch.is_tensor(sample) else sample

                if device.type == "cuda":
                    torch.cuda.synchronize()
                unified_start = time.time()

                _scores, _masks = PC_infer._predict_prompt(images)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                unified_end = time.time()

                total_infer_seconds += (unified_end - unified_start)
                num_proc_images += int(bsz)

                if torch.is_tensor(_scores):
                    _scores = _scores.detach().cpu().numpy().tolist()
                elif isinstance(_scores, np.ndarray):
                    _scores = _scores.tolist()
                else:
                    _scores = list(_scores)

                if torch.is_tensor(_masks):
                    _masks = _masks.detach().cpu().numpy()
                    _masks = [_masks[i] for i in range(_masks.shape[0])]
                elif isinstance(_masks, np.ndarray):
                    _masks = [_masks[i] for i in range(_masks.shape[0])]
                else:
                    _masks = [m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m) for m in _masks]

                scores_task.extend(float(s) for s in _scores)
                masks_task.extend(_masks)

        time_cost = total_infer_seconds / max(1, num_proc_images)

        n = min(len(scores_task), len(labels_gt_task), len(masks_task), len(masks_gt_task))
        scores_task, labels_gt_task = scores_task[:n], labels_gt_task[:n]
        masks_task, masks_gt_task   = masks_task[:n], masks_gt_task[:n]

        anomaly_labels = labels_gt_task
        img_auc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            np.array(scores_task), anomaly_labels
        )["auroc"]
        img_ap = average_precision_score(anomaly_labels, scores_task)

        seg_pred = []
        for m in masks_task:
            m = m if isinstance(m, np.ndarray) else np.asarray(m)
            if m.ndim == 3:
                m = m[..., 0]
            if m.shape != (224, 224):
                m = cv2.resize(m.astype(np.float32), (224, 224))
            else:
                m = m.astype(np.float32)
            seg_pred.append(m)

        masks_arr = []
        for gtm in masks_gt_task:
            g = gtm if isinstance(gtm, np.ndarray) else np.asarray(gtm)
            if g.ndim == 3:
                g = g[..., 0]
            if g.shape != (224, 224):
                g = cv2.resize(g.astype(np.uint8), (224, 224))
            else:
                g = g.astype(np.uint8)
            masks_arr.append(g)

        seg_flat   = np.concatenate([m.reshape(-1) for m in seg_pred])
        masks_flat = np.concatenate([g.reshape(-1) for g in masks_arr])
        pixel_ap   = average_precision_score(masks_flat.astype(int), seg_flat)

        pixel_auc = patchcore.metrics.compute_pixelwise_retrieval_metrics(seg_pred, masks_arr)["auroc"]
        pixel_pro, _ = calculate_au_pro(np.array(masks_arr), np.array(seg_pred))

        result_collect_final.append({
            "dataset_name": getattr(dataloaders["testing"], "name", f"task_{tid_loop}"),
            "instance_auroc": img_auc,
            "full_pixel_auroc": pixel_auc,
            "anomaly_pixel_auroc": pixel_auc,
            "image_ap": img_ap,
            "pixel_ap": pixel_ap,
            "pixel_pro": pixel_pro,
            "time_cost:": time_cost,
        })

        print(
            "current task:{}/test task:{}, image_auc:{:.3f}, pixel_auc:{:.3f}, "
            "image_ap:{:.3f}, pixel_ap:{:.3f}, pixel_pro:{:.3f}, time_cost:{:.3f}".format(
                tid_loop + 1,
                len(list_of_dataloaders),
                img_auc,
                pixel_auc,
                img_ap,
                pixel_ap,
                pixel_pro,
                time_cost,
            )
        )

    metrics_to_avg = [
        "instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc",
        "image_ap", "pixel_ap", "pixel_pro"
    ]
    avg = {m: float(np.mean([r[m] for r in result_collect_final])) for m in metrics_to_avg}
    avg_time = float(np.mean([r.get("time_cost:", r.get("time_cost", 0.0)) for r in result_collect_final]))

    print(
        "Unified (final) averages -> image_auc:{:.3f}, pixel_auc:{:.3f}, anomaly_pixel_auroc:{:.3f}, "
        "image_ap:{:.3f}, pixel_ap:{:.3f}, pixel_pro:{:.3f}, time_cost:{:.3f}".format(
            avg["instance_auroc"], avg["full_pixel_auroc"], avg["anomaly_pixel_auroc"],
            avg["image_ap"], avg["pixel_ap"], avg["pixel_pro"], avg_time
        )
    )

    print('Average result with limited')
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

    print('Average result without limited memory')
    if len(result_collect_nolimit) == 0:
        LOGGER.info("No no-limit results were collected — skipping no-limit summary.")
    else:
        basic_result_metric_names = list(result_collect_nolimit[-1].keys())[1:]
        basic_result_dataset_names = [results["dataset_name"] for results in result_collect_nolimit]
        basic_result_scores = [list(results.values())[1:] for results in result_collect_nolimit]
        patchcore.utils.compute_and_store_final_results(
            run_save_path_nolimit,
            basic_result_scores,
            column_names=basic_result_metric_names,
            row_names=basic_result_dataset_names,
        )


@main.command("ucad")
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
@click.option("--key_size", type=int, default=196, show_default=True)
@click.option("--convnext_variant", type=str, default="convnext_base_in22ft1k", show_default=True)
@click.option("--prompt_inject", type=str, default="all",
              help="Injection spec. 'all' or comma/range list like 's2b1,s3b3,s3b6' or 's3:*' or 's3b1-9'.")
@click.option("--analysis_site", type=str, default="s3b6",
              help="Fixed analysis site for features, e.g., 's3b6'.")
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
    convnext_variant,
    prompt_inject,
    analysis_site,
    key_size
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_score_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
                convnext_variant=convnext_variant,
                prompt_inject=prompt_inject,
                analysis_site=analysis_site,
                key_fps_k=key_size
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)
    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=224, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset
            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }
            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


# ======== PRO metric helpers ========
from sklearn.metrics import average_precision_score

def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)
    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)
        fprs.append(fpr)
        pros.append(pro)
    fprs = fprs[::-1]
    pros = pros[::-1]
    return fprs, pros

def trapezoid(x, y, x_max=None):
    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("WARNING: Non-finite values present in trapezoid computation. Using finite values only.")
    x = x[finite_mask]
    y = y[finite_mask]

    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x.tolist(), x_max)
            assert 0 < ins < len(x)
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        mask = x <= x_max
        x = x[mask]; y = y[mask]

    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction

class GroundTruthComponent:
    def __init__(self, anomaly_scores):
        self.anomaly_scores = np.sort(anomaly_scores.copy())
        self.index = 0
        self.last_threshold = None

    def compute_overlap(self, threshold):
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold
        while self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold:
            self.index += 1
        return 1.0 - self.index / len(self.anomaly_scores)

from scipy.ndimage import label as _label

def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    assert len(anomaly_maps) == len(ground_truth_maps)
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(sum(m.size for m in ground_truth_maps))
    ok_index = 0
    structure = np.ones((3, 3), dtype=int)
    for gt_map, pred_map in zip(ground_truth_maps, anomaly_maps):
        labeled, n_components = _label(gt_map, structure)

        ok_pixels = pred_map[labeled == 0]
        anomaly_scores_ok_pixels[ok_index:ok_index + len(ok_pixels)] = ok_pixels.flatten()
        ok_index += len(ok_pixels)

        for comp_idx in range(1, n_components + 1):
            component_scores = pred_map[labeled == comp_idx]
            ground_truth_components.append(GroundTruthComponent(component_scores.flatten()))

    anomaly_scores_ok_pixels = np.sort(anomaly_scores_ok_pixels[:ok_index])
    return ground_truth_components, anomaly_scores_ok_pixels

def calculate_au_pro(gts, preds, integration_limit=0.3, num_thresholds=100):
    fprs, pros = compute_pro(preds, gts, num_thresholds)
    au_pro = trapezoid(fprs, pros, x_max=integration_limit)
    au_pro /= integration_limit
    return au_pro, (fprs, pros)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
