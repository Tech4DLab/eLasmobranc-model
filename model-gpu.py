import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from django.conf import settings
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from mainpage.modelo.CLIP.clip import clip
from mainpage.modelo.encoded_dict import encoded_dict
from mainpage.modelo.global_means_tensor import global_means_tensor


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resize_with_padding(
    img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Resize an image preserving aspect ratio and pad to target_size.
    """
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = round(new_width / img_ratio)
    else:
        new_height = target_size[1]
        new_width = round(new_height * img_ratio)

    img_resized = img.resize((new_width, new_height), resample=Image.BICUBIC)

    new_img = Image.new("RGB", target_size, fill_color)
    upper_left = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    new_img.paste(img_resized, upper_left)

    return new_img


def collect_images(root_path: str, exclude_classes: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str]]:
    """
    Collect image file paths from:
      - files inside root_path (label='root')
      - files inside subfolders (label=subfolder name)

    exclude_classes: list of substrings; if any substring appears in a folder name (lowercased),
    that folder is skipped.
    """
    exclude_classes = exclude_classes or []
    image_paths: List[str] = []
    labels: List[str] = []

    for item_name in os.listdir(root_path):
        item_path = os.path.join(root_path, item_name)

        if os.path.isfile(item_path):
            ext = os.path.splitext(item_name)[1].lower()
            if ext in VALID_EXTENSIONS:
                image_paths.append(item_path)
                labels.append("root")
            continue

        if os.path.isdir(item_path):
            class_name = item_name
            if any(excl.lower() in class_name.lower() for excl in exclude_classes):
                continue

            for file_name in os.listdir(item_path):
                sub_path = os.path.join(item_path, file_name)
                if not os.path.isfile(sub_path):
                    continue

                ext = os.path.splitext(file_name)[1].lower()
                if ext in VALID_EXTENSIONS:
                    image_paths.append(sub_path)
                    labels.append(class_name)

    return image_paths, labels


def get_imgAtt_multiQ(key_paths: List[torch.Tensor], queries: torch.Tensor) -> torch.Tensor:
    """
    Compute attention-like weights for multiple query vectors across spatial tokens.

    key_paths: list of tensors shaped [num_tokens, dim] (one per image)
    queries: tensor shaped [num_queries, dim]
    returns: tensor shaped [num_images, num_queries, num_tokens]
    """
    with torch.no_grad():
        maps = []
        for feat_v in key_paths:
            scores = torch.matmul(feat_v, queries.T)
            scores = scores.max(dim=0, keepdim=True).values - scores
            a_weight = torch.softmax(scores, dim=-1)
            maps.append(a_weight.T)
        return torch.stack(maps)


def _proj_after_ln_post(x: torch.Tensor, modelCLIP) -> torch.Tensor:
    """
    Apply CLIP visual projection if it exists.
    """
    if getattr(modelCLIP.visual, "proj", None) is not None:
        return x @ modelCLIP.visual.proj
    return x


def _attn_block(TR, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for one CLIP Transformer residual block using its internal attention/MLP.
    """
    x_in = x
    x = TR.ln_1(x_in)

    q, k, v = F.linear(x, TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, -1)
    n, b, d_model = q.shape
    h = TR.attn.num_heads
    d_head = d_model // h
    scale = d_head**-0.5

    q = (q * scale).contiguous().view(n, b, h, d_head).permute(1, 2, 0, 3)
    k = k.contiguous().view(n, b, h, d_head).permute(1, 2, 0, 3)
    v = v.contiguous().view(n, b, h, d_head).permute(1, 2, 0, 3)

    attn = torch.matmul(q, k.transpose(-2, -1)).softmax(-1)
    out = torch.matmul(attn, v).permute(2, 0, 1, 3).contiguous().view(n, b, d_model)
    out = F.linear(out, TR.attn.out_proj.weight, TR.attn.out_proj.bias)

    x = x_in + out
    x = x + TR.mlp(TR.ln_2(x))
    return x


def _attn_block_with_boost(TR, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
    """
    Same as _attn_block but adds a bias to attention logits before softmax.
    key_bias is broadcasted into attention logits.
    """
    x_in = x
    x = TR.ln_1(x_in)

    q, k, v = F.linear(x, TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, -1)
    n, b, d_model = q.shape
    h = TR.attn.num_heads
    d_head = d_model // h
    scale = d_head**-0.5

    q = (q * scale).contiguous().view(n, b, h, d_head).permute(1, 2, 0, 3)
    k = k.contiguous().view(n, b, h, d_head).permute(1, 2, 0, 3)
    v = v.contiguous().view(n, b, h, d_head).permute(1, 2, 0, 3)

    attn = torch.matmul(q, k.transpose(-2, -1))
    attn = attn + key_bias
    attn = attn.softmax(-1)

    out = torch.matmul(attn, v).permute(2, 0, 1, 3).contiguous().view(n, b, d_model)
    out = F.linear(out, TR.attn.out_proj.weight, TR.attn.out_proj.bias)

    x = x_in + out
    x = x + TR.mlp(TR.ln_2(x))
    return x


def _select_layers(n_layers: int, last_n: Optional[int] = None, indices: Optional[Sequence[int]] = None) -> set:
    """
    Choose which transformer layers will receive the 'boost'.
    """
    if indices is not None:
        return {i for i in indices if 0 <= i < n_layers}

    if last_n is not None:
        last_n = max(0, min(last_n, n_layers))
        return set(range(n_layers - last_n, n_layers))

    return set(range(n_layers))


def normalize_maps(a: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalize along dim=1 (token dimension in your usage).
    """
    a_min = a.min(dim=1, keepdim=True).values
    a_max = a.max(dim=1, keepdim=True).values
    return (a - a_min) / (a_max - a_min + 1e-8)


def replicator_clip_boost(
    x_img: torch.Tensor,
    alpha_w: float,
    invert: bool = False,
    last_n: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    clip_inres: int = 224,
    clip_ksize: Tuple[int, int] = (14, 14),
    modelCLIP=None,
) -> torch.Tensor:
    """
    Run CLIP vision transformer but "boost" attention keys using a token-wise bias built
    from similarities against global_means_tensor.

    NOTE:
    - This keeps your original behavior (half precision conv1 path).
    - If you need CPU support, ensure conv1 input dtype matches model weights (usually float32).
    """
    device = next(modelCLIP.parameters()).device
    x_img = x_img.to(device)

    x = modelCLIP.visual.conv1(x_img.half())
    fea_h, fea_w = x.shape[-2:]

    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls_tok = modelCLIP.visual.class_embedding.to(x.dtype)
    x = torch.cat([cls_tok + torch.zeros(x.size(0), 1, x.size(-1), device=x.device, dtype=x.dtype), x], 1)

    pos = modelCLIP.visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos[:1], pos[1:]

    pos_h, pos_w = clip_inres // clip_ksize[0], clip_inres // clip_ksize[1]
    img_pos = img_pos.view(1, pos_h, pos_w, -1).permute(0, 3, 1, 2)
    img_pos = F.interpolate(img_pos, (fea_h, fea_w), mode="bicubic", align_corners=False)
    img_pos = img_pos.view(1, img_pos.size(1), -1).permute(0, 2, 1)

    pos_full = torch.cat([tok_pos[None, ...], img_pos], 1)
    x = modelCLIP.visual.ln_pre(x + pos_full)

    x = x.permute(1, 0, 2)

    blocks = modelCLIP.visual.transformer.resblocks
    sel = _select_layers(len(blocks), last_n=last_n, indices=indices)

    sel_sorted = sorted(sel)
    first_sel = sel_sorted[0] if sel_sorted else len(blocks)

    for i in range(first_sel):
        x = _attn_block(blocks[i], x)

    x_common = x.clone()

    # ----------------------------
    # Branch A: compute token weights
    # ----------------------------
    x_normal = x_common
    for i in range(first_sel, len(blocks)):
        x_normal = _attn_block(blocks[i], x_normal)

    x_normal = modelCLIP.visual.ln_post(x_normal.permute(1, 0, 2))
    x_normal = _proj_after_ln_post(x_normal, modelCLIP)

    spatial_img_features = [t[1:, :] for t in x_normal]  # drop CLS token for spatial tokens

    gt = torch.as_tensor(global_means_tensor, dtype=torch.float16, device=device)
    a_weight = get_imgAtt_multiQ(spatial_img_features, gt)

    a_weight_max = a_weight.max(dim=1).values
    a_weight_max_norm = normalize_maps(a_weight_max)
    w_trans = a_weight_max_norm.pow(2) * alpha_w

    w = w_trans.view(modelCLIP.visual.conv1(x_img.half()).shape[0], -1)

    if invert:
        w = 1.0 - w

    w_full = torch.cat([torch.zeros(x.size(1), 1, device=x.device, dtype=x.dtype), w.to(x.dtype)], dim=1)
    key_bias = w_full[:, None, None, :]

    # ----------------------------
    # Branch B: run boosted transformer
    # ----------------------------
    x_boost = x_common
    for i in range(first_sel, len(blocks)):
        if i in sel:
            x_boost = _attn_block_with_boost(blocks[i], x_boost, key_bias)
        else:
            x_boost = _attn_block(blocks[i], x_boost)

    x_boost = modelCLIP.visual.ln_post(x_boost.permute(1, 0, 2))
    x_boost = _proj_after_ln_post(x_boost, modelCLIP)
    return x_boost


def build_taxonomy_df() -> pd.DataFrame:
    """
    Build and return the taxonomy table used by the hierarchical classifier.
    """
    data = [
        ["bocanegra", "galeus melastomus", "blackmouth catshark", "carcharhiniformes", "scyliorhinidae", "shark"],
        ["cazon", "galeorhinus galeus", "tope shark", "carcharhiniformes", "triakidae", "shark"],
        ["cerdo_marino", "oxynotus centrina", "angular roughshark", "squaliformes", "oxynotidae", "shark"],
        ["musola", "mustelus mustelus", "smouth-hound", "carcharhiniformes", "triakidae", "shark"],
        ["pintarroja", "scyliorhinus canicula", "small-spotted catshark", "carcharhiniformes", "scyliorhinidae", "shark"],
        ["santiaguesa", "raja undulata", "undulate ray", "rajiformes", "rajidae", "stingray"],
        ["tembladera", "torpedo marmorata", "spotted torpedo", "torpediniformes", "torpedinidae", "stingray"],
    ]
    df = pd.DataFrame(data, columns=["Our_Name", "Scientific Name", "Common Name", "Order", "Family", "Animal"])
    df_clean = df[["Animal", "Order", "Family", "Our_Name"]]
    return df_clean.sort_values(by=["Animal", "Order", "Family", "Our_Name"]).reset_index(drop=True)


def normalize_encoded_dict_inplace(device: torch.device) -> None:
    """
    Ensure each encoded_dict[key][i]["text_embedding"] is:
      - torch.Tensor
      - float16
      - on device
      - L2-normalized
    """
    for key in encoded_dict:
        for i, item in enumerate(encoded_dict[key]):
            emb = item["text_embedding"]
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float16)
            else:
                emb = emb.to(dtype=torch.float16)

            emb = emb.to(device).flatten()
            emb = emb / emb.norm()
            encoded_dict[key][i]["text_embedding"] = emb


def _to_numpy_1d(x) -> List[float]:
    """
    Convert tensor/array-like to a 1D python list of floats.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float().numpy().reshape(-1)
        return arr.tolist()

    import numpy as np

    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr.tolist()


def _mean_embeddings(emb_list: Sequence) -> List[float]:
    """
    Compute mean embedding from a list of vectors (tensor/array-like) and return as list[float].
    """
    import numpy as np

    arr = np.stack([_to_numpy_1d(e) for e in emb_list], axis=0)
    return arr.mean(axis=0).tolist()


def build_esquema_dict() -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Build the 'esquema_dict' structure:
      - species embeddings
      - family mean embeddings
      - order mean embeddings
    """
    species_order = ["bocanegra", "cazon", "cerdo_marino", "musola", "pintarroja", "santiaguesa", "tembladera"]

    taxonomy = {
        "bocanegra": ("carcharhiniformes", "scyliorhinidae"),
        "pintarroja": ("carcharhiniformes", "scyliorhinidae"),
        "cazon": ("carcharhiniformes", "triakidae"),
        "musola": ("carcharhiniformes", "triakidae"),
        "cerdo_marino": ("squaliformes", "oxynotidae"),
        "santiaguesa": ("rajiformes", "rajidae"),
        "tembladera": ("torpediniformes", "torpedinidae"),
    }

    species_to_emb = {sp: emb for sp, emb in zip(species_order, global_means_tensor)}

    family_to_species: Dict[Tuple[str, str], List[str]] = {}
    order_to_species: Dict[str, List[str]] = {}

    for sp in species_to_emb:
        order, family = taxonomy[sp]
        family_to_species.setdefault((order, family), []).append(sp)
        order_to_species.setdefault(order, []).append(sp)

    esquema_dict: Dict[str, Dict[str, List[List[float]]]] = {}

    for sp, emb in species_to_emb.items():
        order, family = taxonomy[sp]
        esquema_dict[sp] = {"group": f"Our_Name_{family}", "embedding": [_to_numpy_1d(emb)]}

    for (order, family), sps in family_to_species.items():
        fam_embs = [species_to_emb[sp] for sp in sps]
        esquema_dict[family] = {"group": f"Family_{order}", "embedding": [_mean_embeddings(fam_embs)]}

    for order, sps in order_to_species.items():
        ord_embs = [species_to_emb[sp] for sp in sps]
        esquema_dict[order] = {"group": f"Order_{order}", "embedding": [_mean_embeddings(ord_embs)]}

    return esquema_dict


def load_binary_efficientnet(device: torch.device) -> nn.Module:
    """
    Load the 'Elasmobranch vs other' EfficientNet model.
    """
    modelEF = efficientnet_b0(weights=None)

    in_features = modelEF.classifier[1].in_features
    modelEF.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Linear(256, 1),
    )

    model_path = Path(settings.BASE_DIR) / "mainpage" / "modelo" / "best_model.pt"
    state = torch.load(model_path, map_location=device)
    modelEF.load_state_dict(state, strict=True)

    modelEF = modelEF.to(device)
    modelEF.eval()
    return modelEF


def main(ruta: str) -> List[List[Optional[str]]]:
    """
    Main entry point (your old `model(ruta)`), refactored:
      1) Collect images from folder
      2) Run binary EfficientNet (elasmobranch vs other)
      3) Run CLIP embeddings with attention boost
      4) Hierarchical taxonomy decision (Animal -> Order -> Family -> Our_Name)
      5) Merge results with 'other'
    """
    image_paths, _ = collect_images(ruta, exclude_classes=[])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP
    modelCLIP, preprocessCLIP = clip.load("ViT-L/14", str(device))

    # Load and resize images
    images: List[Image.Image] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = resize_with_padding(img)
        images.append(img)

    # ----------------------------
    # Binary model (elasmobranch vs other)
    # ----------------------------
    preprocessEF = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    images_tensor_ef = torch.stack([preprocessEF(img) for img in images]).to(device)

    modelEF = load_binary_efficientnet(device)

    with torch.no_grad():
        logits = modelEF(images_tensor_ef).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()  # 1 => other, 0 => elasmobranch (per your later merge logic)

    # ----------------------------
    # CLIP embeddings (for all images, as in your current code)
    # ----------------------------
    images_tensor_clip = torch.stack([preprocessCLIP(img) for img in images]).to(device)

    clip_inres = modelCLIP.visual.input_resolution
    clip_ksize = modelCLIP.visual.conv1.kernel_size

    batch_size = 8
    all_embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, images_tensor_clip.size(0), batch_size):
            batch = images_tensor_clip[i : i + batch_size]

            img_emb = replicator_clip_boost(
                batch,
                alpha_w=1.4,
                invert=False,
                indices=None,
                last_n=1,
                clip_inres=clip_inres,
                clip_ksize=clip_ksize,
                modelCLIP=modelCLIP,
            )

            # Normalize and keep CLS token embedding
            img_emb = F.normalize(img_emb, dim=-1)
            all_embeddings.append(img_emb[:, 0])

    img_embedding = torch.cat(all_embeddings, dim=0)

    # ----------------------------
    # Taxonomy resources
    # ----------------------------
    df_taxonomy = build_taxonomy_df()
    esquema_dict = build_esquema_dict()
    normalize_encoded_dict_inplace(device)

    # ----------------------------
    # Hierarchical classification
    # ----------------------------
    levels = ["Animal", "Order", "Family", "Our_Name"]
    animal_labels = ["shark", "stingray"]
    alphas = [0.1, 0.5, 0.7]  # Animal->Order, Order->Family, Family->Our_Name

    f_results: List[List[str]] = []

    with torch.no_grad():
        for emb in img_embedding:
            result: List[str] = []

            img_norm = emb / emb.norm(dim=-1, keepdim=True)
            dtype = img_norm.dtype

            shark_emb = torch.as_tensor(encoded_dict["shark"][0]["text_embedding"], device=device, dtype=dtype).flatten()
            sting_emb = torch.as_tensor(encoded_dict["stingray"][0]["text_embedding"], device=device, dtype=dtype).flatten()

            sims_animal = (img_norm @ torch.stack([shark_emb, sting_emb]).T).squeeze(0)
            result.append(animal_labels[sims_animal.argmax().item()])

            i = 0
            while i < len(levels) - 1:
                next_level = (
                    df_taxonomy.loc[df_taxonomy[levels[i]] == result[-1], levels[i + 1]].unique().tolist()
                )

                # If only one option exists, take it directly
                if len(next_level) == 1:
                    result.append(next_level[0])
                    i += 1
                    continue

                # Text embeddings for candidates
                embs_text = torch.stack(
                    [
                        torch.as_tensor(encoded_dict[k][0]["text_embedding"], device=device, dtype=dtype).flatten()
                        for k in next_level
                    ]
                )

                # Schematic embeddings for candidates
                embs_esq = torch.stack(
                    [
                        torch.as_tensor(esquema_dict[k]["embedding"], device=device, dtype=dtype).flatten()
                        for k in next_level
                    ]
                )

                sims_text = (img_norm @ embs_text.T).squeeze(0)
                sims_esq = (img_norm @ embs_esq.T).squeeze(0)

                alpha = alphas[i]
                sims = alpha * sims_text + (1.0 - alpha) * sims_esq

                result.append(next_level[sims.argmax().item()])
                i += 1

            f_results.append(result)

    # ----------------------------
    # Merge with binary predictions
    # ----------------------------
    merged_results: List[List[Optional[str]]] = []
    for i, p in enumerate(preds):
        if p.item() == 0:
            merged_results.append(f_results[i])
        else:
            merged_results.append(["other", None, None, None])

    return merged_results
