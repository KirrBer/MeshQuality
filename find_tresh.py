"""
Подбор оптимальных порогов для каждого дефекта и quality
на OOF-предсказаниях (out-of-fold) по 5 фолдам.

Каждая из 5 обученных моделей применяется ТОЛЬКО к своей val-части,
поэтому пороги подбираются без утечки данных.

Запуск:
    python find_thresholds.py \
        --train_csv train.csv \
        --train_dir train/ \
        --mesh_dir train/ \
        --model_pattern "best_model_fold{}.pth" \
        --output thresholds.json
"""

import os
import json
import random
import argparse
import zlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score

# ==================== КОНСТАНТЫ ====================
SEED = 42
DEFECT_NAMES = [
    'abstract', 'artifacts', 'intersection', 'lowpoly', 'noisy',
    'open', 'partial', 'scale', 'set', 'simple'
]
NUM_POINTS = 1024
FOLD_SEEDS = [42, 2024, 418, 122, 1756]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


def worker_init_fn(worker_id):
    ws = torch.initial_seed() % (2 ** 31)
    np.random.seed(ws)
    random.seed(ws)
    torch.manual_seed(ws)


# ==================== ОБРАБОТКА ТОЧЕК ====================
def load_and_normalize_vertices(npz_path, item_id):
    if not os.path.exists(npz_path):
        return None
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if 'vertices' not in data:
                return None
            vertices = np.asarray(data['vertices'], dtype=np.float32)
        if len(vertices) == 0:
            return None
        centroid = vertices.mean(axis=0)
        vertices = vertices - centroid
        scale = np.max(np.linalg.norm(vertices, axis=1)) + 1e-8
        vertices = vertices / scale
        return vertices
    except Exception:
        return None


def sample_points(vertices, num_points, rng):
    if vertices is None or len(vertices) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    replace = len(vertices) < num_points
    idxs = rng.choice(len(vertices), num_points, replace=replace)
    return np.ascontiguousarray(vertices[idxs], dtype=np.float32)


# ==================== POINTNET ====================
class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        self.tnet_conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.tnet_fc = nn.Sequential(
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, input_dim * input_dim),
        )

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(feature_dim)

        self.attn = nn.Sequential(
            nn.Conv1d(feature_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 1, 1),
        )
        self.post = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim), nn.ReLU(), nn.Dropout(0.3),
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        b = x.size(0)
        t = self.tnet_conv(x)
        t = torch.max(t, 2, keepdim=False)[0]
        t_mat = self.tnet_fc(t).view(b, 3, 3)
        identity = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
        t_mat = t_mat + identity
        x = torch.bmm(t_mat, x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        max_feat = torch.max(x, 2, keepdim=False)[0]
        attn_w = torch.softmax(self.attn(x), dim=2)
        attn_feat = (x * attn_w).sum(dim=2)
        combined = torch.cat([max_feat, attn_feat], dim=1)
        return self.post(combined)


# ==================== ВИЗУАЛЬНЫЙ ЭНКОДЕР ====================
class MultiViewClassifier(nn.Module):
    def __init__(self, num_views=6, num_defects=10, backbone='resnet50'):
        super().__init__()
        self.num_views = num_views
        self.num_defects = num_defects

        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            feature_dim = 2048
        elif backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1')
            feature_dim = 1536
        elif backbone == 'vit_b_16':
            self.backbone = models.vit_b_16(weights='IMAGENET1K_V1')
            feature_dim = 768
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        if 'resnet' in backbone:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif 'efficientnet' in backbone:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif 'vit' in backbone:
            self.backbone.heads = nn.Identity()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.view_attention = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.Tanh(), nn.Linear(128, 1),
        )
        self.visual_aggregator = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, views):
        b, v = views.shape[0], views.shape[1]
        views_flat = views.view(b * v, 3, 224, 224)
        features = self.backbone(views_flat)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        features = features.view(b, v, -1)
        attn_scores = self.view_attention(features).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attended = (features * attn_weights.unsqueeze(-1)).sum(dim=1)
        return self.visual_aggregator(attended)


class GatedFusion(nn.Module):
    def __init__(self, visual_dim=512, point_dim=512, out_dim=256):
        super().__init__()
        combined_dim = visual_dim + point_dim
        self.same_dim = (visual_dim == point_dim)
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2), nn.ReLU(),
            nn.Linear(combined_dim // 2, 2), nn.Softmax(dim=-1),
        )
        proj_in = visual_dim + combined_dim if self.same_dim else combined_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(proj_in, out_dim),
            nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(0.4),
        )

    def forward(self, visual, point):
        combined = torch.cat([visual, point], dim=1)
        weights = self.gate(combined)
        if self.same_dim:
            fused = weights[:, 0:1] * visual + weights[:, 1:2] * point
            proj_in = torch.cat([fused, combined], dim=1)
        else:
            fused = torch.cat([weights[:, 0:1] * visual,
                               weights[:, 1:2] * point], dim=1)
            proj_in = torch.cat([fused, combined], dim=1)
        return self.proj(proj_in)


class MeshAwareModel(nn.Module):
    def __init__(self, num_views=6, num_defects=10, backbone='resnet50',
                 pointnet_feature_dim=512):
        super().__init__()
        self.num_defects = num_defects
        self.visual_encoder = MultiViewClassifier(num_views, num_defects, backbone)
        self.pointnet_encoder = PointNetEncoder(3, pointnet_feature_dim)
        self.fusion = GatedFusion(512, pointnet_feature_dim, 256)
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_defects)])
        self.final_head = nn.Linear(256 + num_defects, 1)

    def forward(self, views, points):
        v = self.visual_encoder(views)
        p = self.pointnet_encoder(points)
        combined = self.fusion(v, p)
        defect_outputs = [head(combined).squeeze(-1) for head in self.heads]
        defect_stack = torch.stack(defect_outputs, dim=1)
        quality_input = torch.cat([combined, defect_stack], dim=1)
        quality_output = self.final_head(quality_input).squeeze(-1)
        return defect_outputs, quality_output


# ==================== ДАТАСЕТ ДЛЯ ИНФЕРЕНСА ====================
class InferenceDataset(Dataset):
    """Принимает data_frame + indices напрямую, чтобы можно было
    оценивать любую подвыборку train'а."""

    def __init__(self, data_frame, data_dir, mesh_dir=None,
                 num_points=NUM_POINTS, indices=None):
        self.data = data_frame.copy()
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)
        self.data_dir = data_dir
        self.mesh_dir = mesh_dir
        self.num_points = num_points
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_id = self.data.iloc[idx]['item_id']
        img_path = os.path.join(self.data_dir, f"{item_id}.png")
        img = Image.open(img_path).convert('RGB')

        w, h = img.size
        cw, ch = w // 3, h // 2

        views = []
        for row in range(2):
            for col in range(3):
                left, top = col * cw, row * ch
                crop = img.crop((left, top, left + cw, top + ch))
                views.append(self.transform(crop))
        views = torch.stack(views)

        if self.mesh_dir:
            npz_path = os.path.join(self.mesh_dir, f"{item_id}.npz")
            vertices = load_and_normalize_vertices(npz_path, item_id)
        else:
            vertices = None

        item_seed = zlib.crc32(str(item_id).encode()) % (2 ** 31)
        item_rng = np.random.RandomState(item_seed)
        points = sample_points(vertices, self.num_points, item_rng)
        points = torch.tensor(np.ascontiguousarray(points, dtype=np.float32),
                              dtype=torch.float32)

        return views, points, item_id


# ==================== TTA ====================
def _scale(v, factor):
    B, V, C, H, W = v.shape
    nh, nw = int(round(H * factor)), int(round(W * factor))
    x = F.interpolate(v.view(B * V, C, H, W), size=(nh, nw),
                      mode='bilinear', align_corners=False)
    if factor >= 1.0:
        top, left = (nh - H) // 2, (nw - W) // 2
        x = x[:, :, top:top + H, left:left + W]
    else:
        ph, pw = (H - nh) // 2, (W - nw) // 2
        x = F.pad(x, (pw, W - nw - pw, ph, H - nh - ph), mode='replicate')
    return x.view(B, V, C, H, W)


def _shift(v, dx, dy):
    return torch.roll(v, shifts=(dy, dx), dims=(-2, -1))


def build_tta_transforms(use_tta: bool):
    lst = [lambda v: v]
    if not use_tta:
        return lst
    lst += [
        lambda v: torch.flip(v, dims=[-1]),
        lambda v: torch.flip(v, dims=[-2]),
        lambda v: torch.flip(v, dims=[-1, -2]),
        lambda v: _scale(v, 1.10),
        lambda v: _scale(v, 0.90),
        lambda v: _shift(v, dx=16, dy=0),
        lambda v: _shift(v, dx=0, dy=16),
        lambda v: _shift(v, dx=-16, dy=0),
    ]
    return lst


# ==================== ОБРАБОТКА ОДНОЙ МОДЕЛИ ====================
def load_model(path, device, backbone='resnet50', pointnet_feature_dim=512):
    model = MeshAwareModel(
        num_views=6, num_defects=len(DEFECT_NAMES),
        backbone=backbone, pointnet_feature_dim=pointnet_feature_dim,
    ).to(device)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    else:
        state = ckpt
    if list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_with_model(model, dataloader, device, use_tta=True):
    tta_transforms = build_tta_transforms(use_tta)
    all_ids, all_def, all_q = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            views, points, ids = batch
            views = views.to(device)
            points = points.to(device)

            d_sum, q_sum, n_preds = None, None, 0
            for tta in tta_transforms:
                views_aug = tta(views)
                defect_outputs, quality_output = model(views_aug, points)
                d = torch.stack([torch.sigmoid(o) for o in defect_outputs], dim=1)
                q = torch.sigmoid(quality_output)
                if d_sum is None:
                    d_sum, q_sum = d, q
                else:
                    d_sum = d_sum + d
                    q_sum = q_sum + q
                n_preds += 1

            all_ids.extend(ids)
            all_def.append((d_sum / n_preds).cpu().numpy())
            all_q.extend((q_sum / n_preds).cpu().numpy())

    return all_ids, np.vstack(all_def), np.asarray(all_q)


# ==================== СБОР OOF ====================
def build_5fold_splits(n_samples, n_folds=5, seeds=FOLD_SEEDS):
    folds = []
    for fold_idx in range(n_folds):
        seed = seeds[fold_idx]
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_samples)
        fold_size = n_samples // n_folds
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples
        val_idx = perm[val_start:val_end].tolist()
        folds.append(val_idx)
    return folds


def collect_oof_predictions(data_frame, train_dir, mesh_dir, model_pattern,
                            device, batch_size=32, num_workers=4,
                            use_tta=True, verbose=True):
    """Возвращает:
       item_ids : список item_id в порядке OOF
       prob_defects : (N, 10)
       prob_quality : (N,)
       labels_defects : (N, 10)
       labels_quality : (N,)
    """
    n = len(data_frame)
    folds = build_5fold_splits(n, n_folds=5, seeds=FOLD_SEEDS)

    oof_ids = []
    oof_def = []
    oof_q = []
    oof_labels_def = []
    oof_labels_q = []

    defect_cols = [c for c in data_frame.columns if c not in ('item_id', 'quality')]

    for fold_idx, val_idx in enumerate(folds):
        model_path = model_pattern.format(fold_idx)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        if verbose:
            print(f"\n[fold {fold_idx}] модель={model_path} | val={len(val_idx)}")

        model = load_model(model_path, device)

        # Датасет только на val-части этого фолда
        ds = InferenceDataset(data_frame, train_dir, mesh_dir,
                              num_points=NUM_POINTS, indices=val_idx)
        # Сохраняем порядок метки через тот же val_idx
        val_df = data_frame.iloc[val_idx].reset_index(drop=True)
        labels_def = val_df[defect_cols].values.astype(np.float32)
        labels_q = val_df['quality'].values.astype(np.float32)

        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        worker_init_fn=worker_init_fn)

        ids, probs_def, probs_q = predict_with_model(model, dl, device,
                                                     use_tta=use_tta)

        # Проверим порядок: ids должен совпадать с val_df['item_id']
        if list(ids) != list(val_df['item_id'].astype(str)):
            # Если не совпадает — пересоберём в правильном порядке
            id_to_idx = {str(iid): i for i, iid in enumerate(ids)}
            reorder = [id_to_idx[str(iid)] for iid in val_df['item_id']]
            probs_def = probs_def[reorder]
            probs_q = probs_q[reorder]

        oof_ids.extend(val_df['item_id'].tolist())
        oof_def.append(probs_def)
        oof_q.extend(probs_q)
        oof_labels_def.append(labels_def)
        oof_labels_q.extend(labels_q)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_def = np.vstack(oof_def)
    oof_q = np.asarray(oof_q)
    oof_labels_def = np.vstack(oof_labels_def)
    oof_labels_q = np.asarray(oof_labels_q)

    if verbose:
        print(f"\nOOF собрано: {len(oof_ids)} образцов")

    return (oof_ids, oof_def, oof_q, oof_labels_def, oof_labels_q)


# ==================== ПОИСК ПОРОГОВ ====================
def find_best_threshold(y_true, y_prob, grid=np.arange(0.05, 0.96, 0.01)):
    """Brute force по сетке. Возвращает (best_thr, best_f1, all_f1)."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        # Нет позитивов — вернём дефолт
        return 0.5, 0.0, None

    best_thr, best_f1 = 0.5, -1.0
    all_f1 = []
    for thr in grid:
        y_pred = (y_prob > thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        all_f1.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1), (grid, np.array(all_f1))


def compute_final_score(f1_quality, f1_artefacts):
    return 10.0 * f1_quality + 10.0 * f1_artefacts


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='OOF-подбор порогов')
    parser.add_argument('--train_csv', type=str, default='train.csv')
    parser.add_argument('--train_dir', type=str, default='train/')
    parser.add_argument('--mesh_dir', type=str, default='train/')
    parser.add_argument('--model_pattern', type=str,
                        default='best_model_fold{}.pth',
                        help='Шаблон пути к моделям, {} — номер фолда')
    parser.add_argument('--output', type=str, default='thresholds.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pointnet_feature_dim', type=int, default=512)
    parser.add_argument('--use_tta', dest='use_tta', action='store_true')
    parser.add_argument('--no_tta', dest='use_tta', action='store_false')
    parser.set_defaults(use_tta=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'TTA: {"ON" if args.use_tta else "OFF"}')

    data_frame = pd.read_csv(args.train_csv)
    print(f'Всего в train.csv: {len(data_frame)}')

    # ---------- Сбор OOF-предсказаний ----------
    (oof_ids, oof_def, oof_q,
     labels_def, labels_q) = collect_oof_predictions(
        data_frame=data_frame,
        train_dir=args.train_dir,
        mesh_dir=args.mesh_dir,
        model_pattern=args.model_pattern,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tta=args.use_tta,
        verbose=True,
    )

    # ---------- Поиск лучших порогов ----------
    print("\n" + "=" * 72)
    print("ПОДБОР ПОРОГОВ НА OOF")
    print("=" * 72)

    thresholds = {}
    per_class_info = []

    # Quality
    q_thr, q_f1, _ = find_best_threshold(labels_q, oof_q)
    thresholds['quality'] = q_thr
    print(f"{'quality':14s}  pos={labels_q.mean():.4f}  thr={q_thr:.2f}  F1={q_f1:.4f}")

    # Defects
    for i, name in enumerate(DEFECT_NAMES):
        y = labels_def[:, i]
        p = oof_def[:, i]
        thr, f1, _ = find_best_threshold(y, p)
        thresholds[name] = thr
        per_class_info.append((name, y.mean(), thr, f1))
        print(f"{name:14s}  pos={y.mean():.4f}  thr={thr:.2f}  F1={f1:.4f}")

    # ---------- Итоговый weighted F1 по дефектам ----------
    preds_def = np.zeros_like(labels_def, dtype=int)
    for i, name in enumerate(DEFECT_NAMES):
        preds_def[:, i] = (oof_def[:, i] > thresholds[name]).astype(int)
    f1_artefacts_weighted = f1_score(labels_def, preds_def,
                                     average='weighted', zero_division=0)

    final_score = compute_final_score(q_f1, f1_artefacts_weighted)

    print("\n" + "=" * 72)
    print("ИТОГ OOF")
    print("=" * 72)
    print(f"F1 Quality           : {q_f1:.4f}")
    print(f"F1 Artefacts (weight): {f1_artefacts_weighted:.4f}")
    print(f"FINAL (10*Q + 10*A)  : {final_score:.4f}")
    print("=" * 72)

    # ---------- Сохранение ----------
    out = {
        'quality_threshold': thresholds['quality'],
        'defect_thresholds': {k: thresholds[k] for k in DEFECT_NAMES},
        'oof_metrics': {
            'f1_quality': q_f1,
            'f1_artefacts_weighted': f1_artefacts_weighted,
            'final_score': final_score,
        },
        'per_class': {
            name: {'pos_ratio': float(pos), 'threshold': float(thr), 'f1': float(f1)}
            for name, pos, thr, f1 in per_class_info
        },
        'meta': {
            'n_samples': int(len(oof_ids)),
            'use_tta': bool(args.use_tta),
            'fold_seeds': FOLD_SEEDS,
            'seed': SEED,
        },
    }

    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Пороги сохранены в {args.output}")

    # ---------- Готовый сниппет для test.py ----------
    print("\n" + "=" * 72)
    print("ГОТОВЫЙ СНИППЕТ ДЛЯ test.py")
    print("=" * 72)
    print(f"QUALITY_THRESHOLD = {thresholds['quality']:.2f}\n")
    print("DEFAULT_DEFECT_THRESHOLDS = {")
    for name in DEFECT_NAMES:
        print(f"    '{name}':".ljust(20) + f"{thresholds[name]:.2f},")
    print("}")
    print("=" * 72)


if __name__ == '__main__':
    main()