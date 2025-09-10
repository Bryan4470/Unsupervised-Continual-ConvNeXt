import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import re
from typing import Dict, List, Optional, Tuple

from .dualprompt_utils import farthest_point_sampling


# -----------------------------
# Prompted depthwise conv block
# -----------------------------
class PromptedDWConv(nn.Module):
    def __init__(self, dwconv: nn.Conv2d):
        super().__init__()
        assert isinstance(dwconv, nn.Conv2d) and dwconv.groups == dwconv.in_channels
        self.dw = dwconv
        for p in self.dw.parameters():
            p.requires_grad = False
        self.kernel_prompt = nn.Parameter(torch.zeros_like(self.dw.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.dw.weight + self.kernel_prompt
        return F.conv2d(
            x, w_eff, bias=self.dw.bias,
            stride=self.dw.stride, padding=self.dw.padding, dilation=self.dw.dilation,
            groups=self.dw.groups,
        )


class DualPromptBlock(nn.Module):
    def __init__(self, block: nn.Module, dw_attr: str,
                 base_h: Optional[int] = None, base_w: Optional[int] = None):
        super().__init__()
        self.block = block
        self.dw_attr = dw_attr
        mod = getattr(self.block, self.dw_attr, None)
        if not isinstance(mod, PromptedDWConv):
            raise RuntimeError("DualPromptBlock expects PromptedDWConv on the target depthwise conv.")
        self.mask_prompt: Optional[nn.Parameter] = None
        self.base_size = (base_h, base_w) if (base_h and base_w) else None
        # Learnable scale for FiLM-style gating
        self.mask_scale = nn.Parameter(torch.tensor(1.0))

        # Eager-create mask if base size known
        if self.base_size is not None:
            H, W = self.base_size
            mp = torch.zeros(1, 1, H, W)
            self.mask_prompt = nn.Parameter(mp, requires_grad=True)

    def ensure_mask(self, x: torch.Tensor):
        if self.mask_prompt is None:
            _, _, H, W = x.shape
            if self.base_size is None:
                self.base_size = (H, W)
            mp = torch.zeros(1, 1, *self.base_size, device=x.device)
            self.mask_prompt = nn.Parameter(mp, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.ensure_mask(x)
        if hasattr(self, 'mask_prompt') and self.mask_prompt is not None:
            mask = F.interpolate(self.mask_prompt, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x * (1.0 + self.mask_scale * torch.tanh(mask))
        return self.block(x)

    @property
    def kernel_prompt(self) -> nn.Parameter:
        return getattr(self.block, self.dw_attr).kernel_prompt


# -----------------------------------------
# ConvNeXt Dual Prompt Backbone
# -----------------------------------------
class ConvNeXtDualPrompt(nn.Module):
    def __init__(self,
        variant: str = "convnext_small",
        pretrained: bool = True,
        proj_dim: int = 768,
        # EITHER provide explicit sites...
        injection_sites: Optional[List[Tuple[int, int]]] = None,
        # ...OR a string spec you can sweep with
        injection_spec: Optional[str] = None,
        # Single fixed analysis layer (e.g., "s3b6" = Stage 3 Block 6 == indices (2,5))
        analysis_site: Optional[str] = "s3b6",
        key_fps_k: int = 128,
        temperature: float = 0.5,
        base_input_hw: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        # Base ConvNeXt
        self.model = timm.create_model(variant, pretrained=pretrained)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.has_stem = hasattr(self.model, "stem")
        self.stages = getattr(self.model, "stages", None)
        if self.stages is None:
            raise RuntimeError("ConvNeXtDualPrompt expects timm ConvNeXt with a 'stages' attribute.")

        # ---- NEW: parse analysis site & injection spec ----
        self.analysis_stage_idx = self._parse_site_str(analysis_site or "s3b6")[0]


        print("analysis sites >>>>>>>>>>", self.analysis_stage_idx)

        if injection_spec is not None and injection_sites is None:
            self.injection_sites = self._parse_injection_spec(injection_spec)
            print("parsed injection sites for >>>>>>>>>>", injection_spec)
        else:
            print("default injection sites>>>>>>>>>>")
            # default
            self.injection_sites = injection_sites or [(1, 0), (2, 2), (2, 5)]

        self.base_input_hw = base_input_hw
        self._attach_dual_prompts()

        self._stage_channels = self._infer_stage_channels_via_forward()
        c_analysis = self._stage_channels[self.analysis_stage_idx]
        c_stage4   = self._stage_channels[3]

        # Projection from analysis stage (single fixed analysis layer)
        # c_analysis = self._stage_out_channels(self.analysis_stage_idx)
        self.seg_proj = nn.Conv2d(c_analysis, proj_dim, kernel_size=1)

        # Classification head on final stage
        # c_stage4 = self._stage_out_channels(3)
        self.cls_proj = nn.Linear(c_stage4, proj_dim) if c_stage4 != proj_dim else nn.Identity()

        self.cls_prototypes: Dict[int, torch.Tensor] = {}
        self.seg_prototypes: Dict[int, torch.Tensor] = {}
        self.keys_by_task: Dict[int, torch.Tensor] = {}
        self.key_fps_k = key_fps_k

        self.temperature = temperature
        self.use_e_prompt = True   # for SAM supervision

        print("key_fps_k>>>>>>>>>>>>", self.key_fps_k)

    # ---------- parsing helpers ----------
    def _parse_site_str(self, token: str) -> Tuple[int, int]:
        m = re.fullmatch(r"s(\d+)b(\d+)", token.lower().strip())
        if not m:
            raise ValueError(f"Bad site token '{token}'. Use like 's3b6'.")
        s = int(m.group(1)) - 1
        b = int(m.group(2)) - 1
        return s, b

    def _parse_injection_spec(self, spec: str) -> List[Tuple[int, int]]:
        print("initial parse: ", spec)
        spec = spec.lower().strip()
        sites: List[Tuple[int, int]] = []
        if spec in ("all", "*"):
            for s_idx, stage in enumerate(self.stages):
                for b_idx, _blk in enumerate(stage.blocks):
                    if self._find_depthwise_attr(_blk) is not None:
                        sites.append((s_idx, b_idx))
            return sites

        parts = [p.strip() for p in spec.split(",") if p.strip()]
        for p in parts:
            m_star = re.fullmatch(r"s(\d+):\*", p)  # sK:*  → all blocks in stage K
            if m_star:
                s_idx = int(m_star.group(1)) - 1
                for b_idx, _blk in enumerate(self.stages[s_idx].blocks):
                    if self._find_depthwise_attr(_blk) is not None:
                        sites.append((s_idx, b_idx))
                continue

            m_rng = re.fullmatch(r"s(\d+)b(\d+)-(\d+)", p)  # sK bI-J  → range
            if m_rng:
                s_idx = int(m_rng.group(1)) - 1
                i0 = int(m_rng.group(2)) - 1
                i1 = int(m_rng.group(3)) - 1
                i0, i1 = min(i0, i1), max(i0, i1)
                for b in range(i0, i1 + 1):
                    blk = self.stages[s_idx].blocks[b]
                    if self._find_depthwise_attr(blk) is not None:
                        sites.append((s_idx, b))
                continue

            s_idx, b_idx = self._parse_site_str(p)          # single site sK bI
            if self._find_depthwise_attr(self.stages[s_idx].blocks[b_idx]) is None:
                raise RuntimeError(f"No depthwise conv at {p}.")
            sites.append((s_idx, b_idx))

        sites = sorted(set(sites))

        print("parsed values: " + ", ".join(map(str, sites)))
        return sites

    # ---------- model surgery ----------

    def _stage_out_channels(self, stage_idx: int) -> int:
        stage = self.stages[stage_idx]
        blocks = getattr(stage, "blocks", None)
        last = blocks[-1]

        # helper: try to read channels from a block (wrapped or raw)
        def _ch_from_block(m):
            # timm convnext blocks usually have pwconv2 / norm
            if hasattr(m, "pwconv2") and hasattr(m.pwconv2, "out_channels"):
                return m.pwconv2.out_channels
            if hasattr(m, "norm"):
                try:
                    return int(m.norm.normalized_shape[0])
                except Exception:
                    pass
            return None

        # 1) try the block itself
        c = _ch_from_block(last)
        # 2) if wrapped (DualPromptBlock), look at the inner block
        if c is None and hasattr(last, "block"):
            c = _ch_from_block(last.block)
        if c is not None:
            return c

        # 3) FINAL FALLBACK: run a tiny forward to measure the channel dim
        H, W = self.base_input_hw or (224, 224)
        dev = self._dev()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W, device=dev)
            feats = self._forward_stages(dummy)
            return feats[stage_idx].shape[1]

    def _infer_stage_channels_via_forward(self) -> list:
        H, W = self.base_input_hw or (224, 224)
        dev = self._dev()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W, device=dev)
            feats = self._forward_stages(dummy)   # list of (B, C, H, W)
        return [f.shape[1] for f in feats]        # channel dims per stage



    def _find_depthwise_attr(self, blk: nn.Module) -> Optional[str]:
        if hasattr(blk, "dwconv") and isinstance(blk.dwconv, nn.Conv2d):
            return "dwconv"
        for n, m in blk.named_children():
            if isinstance(m, nn.Conv2d) and m.groups == m.in_channels:
                return n
        return None

    def _attach_dual_prompts(self):
        H_in, W_in = (self.base_input_hw or (224, 224))
        stage_stride = {0: 4, 1: 8, 2: 16, 3: 32}

        for s_idx, b_idx in self.injection_sites:
            blk = self.stages[s_idx].blocks[b_idx]
            dw_attr = self._find_depthwise_attr(blk)
            if dw_attr is None:
                raise RuntimeError(f"Cannot find depthwise conv in stages[{s_idx}].blocks[{b_idx}]")
            old_dw = getattr(blk, dw_attr)
            setattr(blk, dw_attr, PromptedDWConv(old_dw))

            # base feature-map size for this stage
            stride = stage_stride[s_idx]
            bh, bw = H_in // stride, W_in // stride

            # wrap
            self.stages[s_idx].blocks[b_idx] = DualPromptBlock(
                blk, dw_attr=dw_attr, base_h=bh, base_w=bw
            )

    # ---------- UCAD API ----------
    def set_prompt_task(self, task_id: int, prompt_state: Dict[str, torch.Tensor]):
        """
        Install prompts (mask & kernel) for all injection sites for a given task.
        Keys per site:
           - 'site_{i}_mask':   (1,1,H,W)
           - 'site_{i}_kernel': (C,1,Kh,Kw)
        """
        dev = self._dev()
        for i, (s_idx, b_idx) in enumerate(self.injection_sites):
            stage = self.stages[s_idx]
            blocks = stage.blocks
            site: DualPromptBlock = blocks[b_idx]
            if f"site_{i}_mask" in prompt_state:
                param = prompt_state[f"site_{i}_mask"].to(dev)
                if site.mask_prompt is None:
                    site.mask_prompt = nn.Parameter(param.clone().detach())
                else:
                    site.mask_prompt.data.copy_(param)
            if f"site_{i}_kernel" in prompt_state:
                kp = prompt_state[f"site_{i}_kernel"].to(dev)
                site.kernel_prompt.data.copy_(kp)

        # restore heads if present
        if "seg_proj_w" in prompt_state:
            self.seg_proj.weight.data.copy_(prompt_state["seg_proj_w"].to(dev))
        if "seg_proj_b" in prompt_state and prompt_state["seg_proj_b"] is not None and self.seg_proj.bias is not None:
            self.seg_proj.bias.data.copy_(prompt_state["seg_proj_b"].to(dev))

        if isinstance(self.cls_proj, nn.Linear):
            if "cls_proj_w" in prompt_state:
                self.cls_proj.weight.data.copy_(prompt_state["cls_proj_w"].to(dev))
            if "cls_proj_b" in prompt_state and prompt_state["cls_proj_b"] is not None and self.cls_proj.bias is not None:
                self.cls_proj.bias.data.copy_(prompt_state["cls_proj_b"].to(dev))

    @torch.no_grad()
    def get_prompt_task(self) -> Dict[str, torch.Tensor]:
        out = {}
        for i, (s_idx, b_idx) in enumerate(self.injection_sites):
            stage = self.stages[s_idx]
            blocks = stage.blocks
            site: DualPromptBlock = blocks[b_idx]
            out[f"site_{i}_kernel"] = site.kernel_prompt.detach().clone().cpu()
            if site.mask_prompt is not None:
                out[f"site_{i}_mask"] = site.mask_prompt.detach().clone().cpu()
            else:
                out[f"site_{i}_mask"] = torch.zeros(1, 1, 1, 1)

        # projection heads
        out["seg_proj_w"] = self.seg_proj.weight.detach().clone().cpu()
        out["seg_proj_b"] = (
            self.seg_proj.bias.detach().clone().cpu()
            if self.seg_proj.bias is not None else None
        )
        if isinstance(self.cls_proj, nn.Linear):
            out["cls_proj_w"] = self.cls_proj.weight.detach().clone().cpu()
            out["cls_proj_b"] = (
                self.cls_proj.bias.detach().clone().cpu()
                if self.cls_proj.bias is not None else None
            )
        return out

    @torch.no_grad()
    def get_cur_prompt(self):
        return self.get_prompt_task()

    def set_prompt_cls(self, task_id: int, prototypes: torch.Tensor):
        self.cls_prototypes[task_id] = prototypes.detach().to(self._dev())

    def set_prompt_seg(self, task_id: int, prototypes: torch.Tensor):
        self.seg_prototypes[task_id] = prototypes.detach().to(self._dev())

    def set_task_keys(self, task_id: int, key_feats: torch.Tensor, m: Optional[int] = None):
        key_feats = key_feats.detach().to(self._dev())
        m = m or self.key_fps_k
        idx = farthest_point_sampling(key_feats, m)
        self.keys_by_task[task_id] = key_feats[idx]

    @torch.no_grad()
    def extract_key_patches(self, x: torch.Tensor, stage_index: Optional[int] = None) -> torch.Tensor:
        """
        Extract mid-layer patch tokens from this instance's backbone.
        Returns (B, N, C).
        """
        dev = self._dev()
        feats = self._forward_stages(x.to(dev))
        si = self.analysis_stage_idx if stage_index is None else stage_index
        fm = feats[si]                   # (B, C, H, W)
        B, C, H, W = fm.shape
        tokens = fm.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return tokens

    def match_task_by_key(self, x: torch.Tensor) -> int:
        assert x.shape[0] == 1
        if not self.keys_by_task:
            raise RuntimeError("No task keys stored.")
        q = self.extract_key_patches(x, stage_index=self.analysis_stage_idx)[0]
        q = F.normalize(q, dim=-1)
        best_tid, best_score = None, float("inf")
        for tid, keys in self.keys_by_task.items():
            k = F.normalize(keys.to(q.device), dim=-1)
            d = torch.cdist(q, k, p=2)                           # (Nq, Nk)
            score = d.min(dim=1).values.mean().item()
            if score < best_score:
                best_score, best_tid = score, tid
        return int(best_tid)

    # ---------- forward ----------
    def _forward_stages(self, x):
        feats = []
        if self.has_stem:
            x = self.model.stem(x)
        for s_idx, stage in enumerate(self.stages):
            if hasattr(stage, "downsample") and stage.downsample is not None:
                x = stage.downsample(x)
            for blk in stage.blocks:
                x = blk(x)
            feats.append(x)
        return feats

    def forward(self, x, train=False, task_id=-1, image_path=None):
        x = x.to(self._dev())
        feats = self._forward_stages(x)
        feat_analysis = feats[self.analysis_stage_idx]
        feat4 = feats[3]

        seg = self.seg_proj(feat_analysis)
        B, C, Ht, Wt = seg.shape
        seg_tokens = seg.flatten(2).transpose(1, 2)

        cls = F.adaptive_avg_pool2d(feat4, 1).flatten(1)
        cls = self.cls_proj(cls)

        if not train:
            return {"seg_feat": [seg_tokens], "pre_logits": cls}

        # --- SAM-supervised contrastive ---
        labels = torch.zeros((B, Ht * Wt), device=seg_tokens.device)
        for i in range(B):
            sam_score = cv2.imread(image_path[i].replace("mvtec2d", "mvtec2d-sam-b"))
            sam_resized = cv2.resize(sam_score, (Ht, Wt))[:, :, 0]
            labels[i] = torch.from_numpy(sam_resized.flatten()).to(seg_tokens.device)

        loss = self._contrastive_loss(seg_tokens, labels, self.temperature)

        return {"loss": loss, "seg_feat": [seg_tokens], "pre_logits": cls}

    # ---------- loss ----------
    def _contrastive_loss(self, features, labels, temperature=0.5):
       
        # features: (B, N, D), labels: (B, N) with equal ids for positives
        f = F.normalize(features, dim=2)
        sim = torch.bmm(f, f.transpose(1, 2)) / temperature            # (B, N, N)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()  # include diagonal
        loss = (-sim * pos_mask + (1.0 - pos_mask) * sim.exp()).mean()
        return loss

    def _dev(self):
        return next(self.parameters()).device

    def forward_features_prompted(self, x: torch.Tensor):
        """
        A helper forward for FLOPs counting.
        Returns the stage-<analysis> projection (seg_feat).
        """
        feats = self._forward_stages(x)
        feat_analysis = feats[self.analysis_stage_idx]
        seg = self.seg_proj(feat_analysis)
        return seg


# ---------- DEBUG ----------
def debug_print_dual_prompt(model):
    print("\n=== DualPrompt injection sites ===")
    # for i, (s_idx, b_idx) in enumerate(model.injection_sites):
    #     site = model.model.stages[s_idx].blocks[b_idx]  # DualPromptBlock
    #     print(f" Site {i}: stages[{s_idx}].blocks[{b_idx}] -> {type(site).__name__}")

    #     dw_attr = getattr(site, "dw_attr", None)
    #     print("   depthwise attr:", dw_attr)

    #     kp_shape = "N/A"
    #     kp_req = "N/A"
    #     if isinstance(dw_attr, str) and hasattr(site.block, dw_attr):
    #         dw_mod = getattr(site.block, dw_attr)  # PromptedDWConv
    #         if hasattr(dw_mod, "kernel_prompt"):
    #             kp_shape = tuple(dw_mod.kernel_prompt.shape)
    #             kp_req = getattr(dw_mod.kernel_prompt, "requires_grad", "N/A")
    #     print("   kernel_prompt shape:", kp_shape, "requires_grad:", kp_req)

    #     has_mask = getattr(site, "mask_prompt", None) is not None
    #     mask_shape = tuple(site.mask_prompt.shape) if has_mask else "N/A"
    #     mask_req = site.mask_prompt.requires_grad if has_mask else "N/A"
    #     print("   mask_prompt exists:", has_mask, "shape:", mask_shape, "requires_grad:", mask_req)

    print("\n=== Parameters of interest (prompts + heads) ===")
    for n, p in model.named_parameters():
        if ("kernel_prompt" in n) or ("mask_prompt" in n) or n.startswith(("seg_proj", "cls_proj")):
            print(f"  {n:80s} {tuple(p.shape)}  requires_grad={p.requires_grad}")

    total = sum(p.numel() for _n, p in model.named_parameters())
    trainable = sum(p.numel() for _n, p in model.named_parameters() if p.requires_grad)
    print(f"\nParam totals: {trainable}/{total} trainable\n")
