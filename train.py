import os
import random
import zlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DEFECT_THRESHOLDS = [
    0.55, 0.35, 0.30, 0.40, 0.35,
    0.40, 0.55, 0.35, 0.50, 0.45,
]
DEFECT_NAMES = [
    'abstract', 'artifacts', 'intersection', 'lowpoly', 'noisy',
    'open', 'partial', 'scale', 'set', 'simple'
]

SEED = 42
NUM_POINTS = 1024


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
    """Пересевает RNG в каждом воркере DataLoader'а."""
    ws = torch.initial_seed() % (2 ** 31)
    np.random.seed(ws)
    random.seed(ws)
    torch.manual_seed(ws)


def load_and_normalize_vertices(npz_path, item_id):
    """
    Читает vertices из .npz и нормализует (центр + scale до единичной сферы).
    Возвращает (N, 3) float32 или None при ошибке.
    """
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
    except Exception as e:
        print(f"  Ошибка чтения {npz_path}: {e}")
        return None


def sample_points(vertices, num_points, rng):
    """Простое случайное семплирование (с повторами при нехватке)."""
    if vertices is None or len(vertices) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    replace = len(vertices) < num_points
    idxs = rng.choice(len(vertices), num_points, replace=replace)
    return np.ascontiguousarray(vertices[idxs], dtype=np.float32)


# =============================================================
#               УСИЛЕННЫЙ POINTNET ENCODER
# =============================================================
class PointNetEncoder(nn.Module):
    """
    Усиленный PointNet:
      - T-Net для выравнивания входного облака точек (3x3 матрица)
      - 5 свёрточных слоёв (64→128→256→512→feature_dim)
      - Совмещённый пулинг: global max + attention pooling
      - Пост-пулинговый MLP для агрегации
    """
    def __init__(self, input_dim=3, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        self.tnet_conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.tnet_fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
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
            nn.Conv1d(feature_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 1, 1),
        )

        self.post = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        batch_size = x.size(0)

        t = self.tnet_conv(x)
        t = torch.max(t, 2, keepdim=False)[0]
        t_mat = self.tnet_fc(t).view(batch_size, 3, 3)
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


class FocalLoss(nn.Module):
    """Focal Loss для борьбы с дисбалансом классов"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -torch.log(p_t + 1e-8)
        loss = alpha_t * focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =============================================================
#                       DATASET
# =============================================================
class MeshDataset(Dataset):
    """
    Датасет без статических геометрических признаков.
    Возвращает (views, points, labels, quality).
    Точки: нормализация + простое случайное семплирование + аугментации.
    """
    def __init__(self, csv_file=None, data_dir='train/', mesh_dir='train/',
                 train=False, indices=None, data_frame=None, num_points=NUM_POINTS):
        self.data = data_frame.copy() if data_frame is not None else pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.mesh_dir = mesh_dir
        self.train = train
        self.num_points = num_points

        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)

        if train:
            self.common_transform = transforms.Compose([
                transforms.RandomApply([transforms.RandomHorizontalFlip(p=1)], p=0.5),
                transforms.RandomApply([transforms.RandomVerticalFlip(p=1)], p=0.4),
                transforms.RandomApply([transforms.RandomRotation(degrees=25)], p=0.6),
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15))], p=0.6),
                transforms.RandomApply([transforms.RandomResizedCrop(size=(300, 300), scale=(0.7, 1.0), ratio=(0.9, 1.1))], p=0.8),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.15)], p=0.8),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0))], p=0.3),
                transforms.RandomApply([transforms.RandomGrayscale(p=1)], p=0.2),
            ])
        else:
            self.common_transform = None

        self.final_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def _augment_points(self, points):
        """Усиленные аугментации точек. Использует глобальный np.random,
        посеянный worker_init_fn (меняется между эпохами)."""
        # 1. Полная 3D-ротация Rz @ Ry @ Rx
        angles = np.random.uniform(-np.pi, np.pi, size=3).astype(np.float32)
        cx, sx = np.cos(angles[0]), np.sin(angles[0])
        cy, sy = np.cos(angles[1]), np.sin(angles[1])
        cz, sz = np.cos(angles[2]), np.sin(angles[2])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        points = points @ (Rz @ Ry @ Rx).T

        # 2. Scale jitter
        points = points * np.float32(np.random.uniform(0.9, 1.1))

        # 3. Gaussian noise
        points = points + np.random.normal(0, 0.01, points.shape).astype(np.float32)

        # 4. Random dropout с восстановлением до num_points
        if np.random.rand() < 0.5:
            drop_ratio = np.random.uniform(0.1, 0.25)
            keep = np.random.rand(len(points)) > drop_ratio
            if keep.sum() > 64:
                points = points[keep]
                if len(points) < self.num_points:
                    n_extra = self.num_points - len(points)
                    extra_idx = np.random.choice(len(points), n_extra, replace=True)
                    points = np.concatenate([points, points[extra_idx]], axis=0)
                elif len(points) > self.num_points:
                    points = points[:self.num_points]

        return points

    def __getitem__(self, idx):
        item_id = self.data.iloc[idx]['item_id']
        img_path = os.path.join(self.data_dir, f"{item_id}.png")
        img = Image.open(img_path).convert('RGB')

        width, height = img.size
        crop_width = width // 3
        crop_height = height // 2

        views = []
        for row in range(2):
            for col in range(3):
                left = col * crop_width
                top = row * crop_height
                right = left + crop_width
                bottom = top + crop_height
                crop = img.crop((left, top, right, bottom))
                if self.common_transform is not None:
                    crop = self.common_transform(crop)
                views.append(self.final_transform(crop))
        views = torch.stack(views)

        defect_columns = [c for c in self.data.columns if c not in ('item_id', 'quality')]
        defect_values = np.array(self.data.iloc[idx][defect_columns].values, dtype=np.float32)
        labels = torch.tensor(defect_values, dtype=torch.float32)
        quality = torch.tensor(self.data.iloc[idx]['quality'], dtype=torch.float32)

        # --- Точки: нормализация + случайное семплирование ---
        npz_path = os.path.join(self.mesh_dir, f"{item_id}.npz")
        vertices = load_and_normalize_vertices(npz_path, item_id)

        # Детерминированный per-item RNG для семплирования (не зависит от воркера/эпохи)
        item_seed = zlib.crc32(str(item_id).encode()) % (2 ** 31)
        item_rng = np.random.RandomState(item_seed)
        points = sample_points(vertices, self.num_points, item_rng)

        # Аугментации (только train)
        if self.train:
            points = self._augment_points(points)

        points = torch.tensor(np.ascontiguousarray(points, dtype=np.float32),
                              dtype=torch.float32)

        return views, points, labels, quality


# =============================================================
#                  ВИЗУАЛЬНЫЙ ЭНКОДЕР (multi-view)
# =============================================================
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

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.view_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.visual_aggregator = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, views):
        batch_size, num_views = views.shape[0], views.shape[1]
        views_flat = views.view(batch_size * num_views, 3, 224, 224)
        features = self.backbone(views_flat)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        features = features.view(batch_size, num_views, -1)
        attn_scores = self.view_attention(features).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attended_features = (features * attn_weights.unsqueeze(-1)).sum(dim=1)
        visual_features = self.visual_aggregator(attended_features)
        return visual_features


# =============================================================
#                    GATED FUSION
# =============================================================
class GatedFusion(nn.Module):
    """
    Взвешенная сумма visual + point с обучаемым gate.
    Работает корректно, когда visual_dim == point_dim (512 == 512).
    """
    def __init__(self, visual_dim=512, point_dim=512, out_dim=256):
        super().__init__()
        combined_dim = visual_dim + point_dim
        self.same_dim = (visual_dim == point_dim)

        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Linear(combined_dim // 2, 2),
            nn.Softmax(dim=-1),
        )

        proj_in = visual_dim + combined_dim if self.same_dim else combined_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(proj_in, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
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


# =============================================================
#                    MESH-AWARE MODEL
# =============================================================
class MeshAwareModel(nn.Module):
    def __init__(self, num_views=6, num_defects=10, backbone='resnet50',
                 pointnet_feature_dim=512):
        super().__init__()
        self.num_defects = num_defects
        self.visual_encoder = MultiViewClassifier(
            num_views=num_views,
            num_defects=num_defects,
            backbone=backbone
        )
        self.pointnet_encoder = PointNetEncoder(input_dim=3, feature_dim=pointnet_feature_dim)
        self.fusion = GatedFusion(
            visual_dim=512,
            point_dim=pointnet_feature_dim,
            out_dim=256,
        )
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_defects)])
        self.final_head = nn.Linear(256 + num_defects, 1)

    def forward(self, views, points):
        visual_features = self.visual_encoder(views)
        point_features = self.pointnet_encoder(points)
        combined = self.fusion(visual_features, point_features)

        defect_outputs = [head(combined).squeeze(-1) for head in self.heads]
        defect_stack = torch.stack(defect_outputs, dim=1)

        quality_input = torch.cat([combined, defect_stack], dim=1)
        quality_output = self.final_head(quality_input).squeeze(-1)

        return defect_outputs, quality_output


# =============================================================
#                    TRAIN / VALIDATE
# =============================================================
def train_epoch(model, dataloader, optimizer, device,
                defect_thresholds=None,
                defect_alphas=None, defect_gammas=None,
                quality_alpha=0.25, quality_gamma=2.0):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_defect_preds = []
    all_defect_labels = []

    if defect_thresholds is None:
        defect_thresholds = DEFECT_THRESHOLDS
    defect_thresholds = np.asarray(defect_thresholds, dtype=np.float32)

    num_defects = len(defect_thresholds)
    if defect_alphas is None:
        defect_alphas = [0.25] * num_defects
    if defect_gammas is None:
        defect_gammas = [2.0] * num_defects
    defect_criteria = [FocalLoss(alpha=defect_alphas[i], gamma=defect_gammas[i])
                       for i in range(num_defects)]
    quality_criterion = FocalLoss(alpha=quality_alpha, gamma=quality_gamma)

    for batch in dataloader:
        views, points, labels, quality = batch
        views = views.to(device)
        points = points.to(device)
        labels = labels.to(device)
        quality = quality.to(device)

        optimizer.zero_grad()
        defect_outputs, quality_output = model(views, points)

        defect_loss = 0.0
        for i, out in enumerate(defect_outputs):
            defect_target = labels[:, i]
            defect_loss += defect_criteria[i](out, defect_target)

        quality_loss = quality_criterion(quality_output, quality)
        loss = defect_loss + 3 * quality_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.detach().item()
        all_preds.extend(torch.sigmoid(quality_output).detach().cpu().numpy())
        all_labels.extend(quality.detach().cpu().numpy())

        defect_preds = torch.stack([torch.sigmoid(out) for out in defect_outputs], dim=1)
        all_defect_preds.append(defect_preds.detach().cpu().numpy())
        all_defect_labels.append(labels.detach().cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    f1_quality = f1_score(all_labels, (all_preds > 0.55).astype(int), zero_division=0)

    all_defect_preds = np.vstack(all_defect_preds)
    all_defect_labels = np.vstack(all_defect_labels)
    all_defect_preds_binary = (all_defect_preds > defect_thresholds).astype(int)
    f1_artefacts_weighted = f1_score(all_defect_labels, all_defect_preds_binary, average='weighted', zero_division=0)

    return total_loss / len(dataloader), f1_quality, f1_artefacts_weighted


def validate_epoch(model, dataloader, device,
                   defect_thresholds=None,
                   defect_alphas=None, defect_gammas=None,
                   quality_alpha=0.25, quality_gamma=2.0,
                   use_optimal_thresholds=True):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_defect_preds = []
    all_defect_labels = []

    if defect_thresholds is None:
        defect_thresholds = DEFECT_THRESHOLDS
    defect_thresholds = np.asarray(defect_thresholds, dtype=np.float32)

    num_defects = len(defect_thresholds)
    if defect_alphas is None:
        defect_alphas = [0.25] * num_defects
    if defect_gammas is None:
        defect_gammas = [2.0] * num_defects
    defect_criteria = [FocalLoss(alpha=defect_alphas[i], gamma=defect_gammas[i])
                       for i in range(num_defects)]
    quality_criterion = FocalLoss(alpha=quality_alpha, gamma=quality_gamma)

    with torch.no_grad():
        for batch in dataloader:
            views, points, labels, quality = batch
            views = views.to(device)
            points = points.to(device)
            labels = labels.to(device)
            quality = quality.to(device)

            defect_outputs, quality_output = model(views, points)

            defect_loss = 0.0
            for i, out in enumerate(defect_outputs):
                defect_target = labels[:, i]
                defect_loss += defect_criteria[i](out, defect_target)

            quality_loss = quality_criterion(quality_output, quality)
            loss = defect_loss + 3 * quality_loss
            total_loss += loss.detach().item()

            all_preds.extend(torch.sigmoid(quality_output).detach().cpu().numpy())
            all_labels.extend(quality.detach().cpu().numpy())

            defect_preds = torch.stack([torch.sigmoid(out) for out in defect_outputs], dim=1)
            all_defect_preds.append(defect_preds.detach().cpu().numpy())
            all_defect_labels.append(labels.detach().cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_defect_preds = np.vstack(all_defect_preds)
    all_defect_labels = np.vstack(all_defect_labels)

    f1_quality_fixed = f1_score(all_labels, (all_preds > 0.55).astype(int), zero_division=0)
    defect_preds_fixed = (all_defect_preds > defect_thresholds).astype(int)
    f1_artefacts_weighted_fixed = f1_score(all_defect_labels, defect_preds_fixed, average='weighted', zero_division=0)

    if use_optimal_thresholds:
        best_quality_thr, best_f1_quality = find_optimal_threshold(all_labels, all_preds)
        f1_quality_opt = best_f1_quality

        defect_thresholds_opt = []
        defect_f1s = []
        for i in range(all_defect_labels.shape[1]):
            best_thr, best_f1 = find_optimal_threshold(all_defect_labels[:, i], all_defect_preds[:, i])
            defect_thresholds_opt.append(best_thr)
            defect_f1s.append(best_f1)
        defect_preds_opt = (all_defect_preds > np.array(defect_thresholds_opt)).astype(int)
        f1_artefacts_weighted_opt = f1_score(all_defect_labels, defect_preds_opt, average='weighted', zero_division=0)

        print(f"  Optimal quality threshold: {best_quality_thr:.3f}, F1: {best_f1_quality:.4f}")
        for i, name in enumerate(DEFECT_NAMES):
            print(f"  {name}: threshold={defect_thresholds_opt[i]:.3f}, F1={defect_f1s[i]:.4f}")

        return (total_loss / len(dataloader),
                f1_quality_opt,
                f1_artefacts_weighted_opt,
                f1_quality_fixed,
                f1_artefacts_weighted_fixed,
                defect_thresholds_opt,
                best_quality_thr)
    else:
        return (total_loss / len(dataloader),
                f1_quality_fixed,
                f1_artefacts_weighted_fixed,
                None, None, None, None)


def find_optimal_threshold(y_true, y_pred, pos_ratio=None):
    if pos_ratio is None:
        pos_ratio = np.mean(y_true)

    thresholds = np.arange(0.2, 0.75, 0.05)
    best_f1 = 0
    best_thr = 0.5

    for thr in thresholds:
        y_pred_binary = (y_pred > thr).astype(int)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    if pos_ratio < 0.2:
        low_thresholds = np.arange(0.05, 0.3, 0.05)
        for thr in low_thresholds:
            y_pred_binary = (y_pred > thr).astype(int)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
    elif pos_ratio > 0.8:
        high_thresholds = np.arange(0.6, 0.95, 0.05)
        for thr in high_thresholds:
            y_pred_binary = (y_pred > thr).astype(int)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

    return best_thr, best_f1


def unfreeze_last_layers(model, num_layers=3):
    if hasattr(model, 'visual_encoder'):
        backbone = model.visual_encoder.backbone
    else:
        backbone = model.backbone

    layer_names = list(backbone.named_children())
    total_layers = len(layer_names)
    backbone.train()
    for param in backbone.parameters():
        param.requires_grad = False

    for i in range(max(0, total_layers - num_layers), total_layers):
        name, layer = layer_names[i]
        for param in layer.parameters():
            param.requires_grad = True
        print(f"✅ Unfrozen layer: {name}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")


# =============================================================
#                    ОБУЧЕНИЕ ОДНОГО ФОЛДА
# =============================================================
def train_single_fold(
    data_frame,
    train_indices, val_indices,
    defect_alphas, defect_gammas,
    quality_alpha, quality_gamma,
    seed, device, save_path,
    stage1_epochs=8, total_epochs=20,
    batch_size=64, num_workers=4,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_dataset = MeshDataset(
        csv_file=None, data_dir='train/', mesh_dir='train/',
        train=True, indices=train_indices, data_frame=data_frame,
    )
    val_dataset = MeshDataset(
        csv_file=None, data_dir='train/', mesh_dir='train/',
        train=False, indices=val_indices, data_frame=data_frame,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    model = MeshAwareModel(
        num_views=6,
        num_defects=len(DEFECT_NAMES),
        backbone='resnet50',
        pointnet_feature_dim=512,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.2
    )

    best_val_f1_final = 0
    best_val_f1_quality = 0
    best_val_f1_artefacts = 0

    def make_train_loader(epoch):
        """Пересоздаём DataLoader каждую эпоху: новый base_seed → разные
        аугментации между эпохами, но воспроизводимо."""
        g = torch.Generator()
        g.manual_seed(seed * 100003 + epoch)
        return DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g,
        )

    # ---------- Этап 1: замороженный backbone ----------
    for epoch in range(stage1_epochs):
        train_loader = make_train_loader(epoch)
        train_loss, train_f1_quality, train_f1_artefacts = train_epoch(
            model, train_loader, optimizer, device,
            defect_thresholds=DEFECT_THRESHOLDS,
            defect_alphas=defect_alphas,
            defect_gammas=defect_gammas,
            quality_alpha=quality_alpha, quality_gamma=quality_gamma,
        )
        val_loss, val_f1_quality_opt, val_f1_artefacts_opt, _, _, _, _ = validate_epoch(
            model, val_loader, device,
            defect_thresholds=DEFECT_THRESHOLDS,
            defect_alphas=defect_alphas,
            defect_gammas=defect_gammas,
            quality_alpha=quality_alpha, quality_gamma=quality_gamma,
            use_optimal_thresholds=True,
        )
        val_f1_final = 10 * val_f1_quality_opt + 10 * val_f1_artefacts_opt
        scheduler.step(val_f1_final)

        print(f"[stage1] Epoch {epoch+1:2d}: "
              f"Train L={train_loss:.4f} Val L={val_loss:.4f} | Val Q={val_f1_quality_opt:.4f} "
              f"A={val_f1_artefacts_opt:.4f} Final={val_f1_final:.4f}")

        if val_f1_final > best_val_f1_final:
            best_val_f1_final = val_f1_final
            best_val_f1_quality = val_f1_quality_opt
            best_val_f1_artefacts = val_f1_artefacts_opt
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best saved -> {save_path}")

    # ---------- Этап 2: разморозка ----------
    checkpoint = torch.load(save_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    unfreeze_last_layers(model, num_layers=3)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.2
    )
    patience_counter = 0

    for epoch in range(stage1_epochs, total_epochs):
        train_loader = make_train_loader(epoch)
        train_loss, train_f1_quality, train_f1_artefacts = train_epoch(
            model, train_loader, optimizer, device,
            defect_thresholds=DEFECT_THRESHOLDS,
            defect_alphas=defect_alphas,
            defect_gammas=defect_gammas,
            quality_alpha=quality_alpha, quality_gamma=quality_gamma,
        )
        val_loss, val_f1_quality_opt, val_f1_artefacts_opt, _, _, _, _ = validate_epoch(
            model, val_loader, device,
            defect_thresholds=DEFECT_THRESHOLDS,
            defect_alphas=defect_alphas,
            defect_gammas=defect_gammas,
            quality_alpha=quality_alpha, quality_gamma=quality_gamma,
            use_optimal_thresholds=True,
        )
        val_f1_final = 10 * val_f1_quality_opt + 10 * val_f1_artefacts_opt
        if val_f1_final < best_val_f1_final:
            patience_counter += 1
            if patience_counter >= 4:
                print("Early stopping triggered.")
                break
        else:
            patience_counter = 0
        scheduler.step(val_f1_final)

        print(f"[stage2] Epoch {epoch+1:2d}: "
              f"Train L={train_loss:.4f} | Val L={val_loss:.4f} | Val Q={val_f1_quality_opt:.4f} "
              f"A={val_f1_artefacts_opt:.4f} Final={val_f1_final:.4f}")

        if val_f1_final > best_val_f1_final:
            best_val_f1_final = val_f1_final
            best_val_f1_quality = val_f1_quality_opt
            best_val_f1_artefacts = val_f1_artefacts_opt
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best saved -> {save_path}")

    return {
        'seed': seed,
        'best_f1_final': best_val_f1_final,
        'best_f1_quality': best_val_f1_quality,
        'best_f1_artefacts': best_val_f1_artefacts,
        'save_path': save_path,
    }


def build_5fold_splits(n_samples, n_folds=5, seeds=(42, 2024, 418, 122, 1756)):
    assert len(seeds) == n_folds
    folds = []
    for fold_idx in range(n_folds):
        seed = seeds[fold_idx]
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_samples)
        fold_size = n_samples // n_folds
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples
        val_idx = perm[val_start:val_end].tolist()
        train_idx = np.concatenate([perm[:val_start], perm[val_end:]]).tolist()
        folds.append((train_idx, val_idx, seed))
    return folds


def main():
    print("Загрузка данных...")
    data_frame = pd.read_csv('train.csv')

    defect_columns = [col for col in data_frame.columns if (col != 'item_id' and col != 'quality')]
    quality_pos_ratio = data_frame['quality'].mean()
    defect_pos_ratios = data_frame[defect_columns].mean().values

    quality_alpha = max(0.05, 1.0 - quality_pos_ratio - 0.1)
    defect_alphas = [max(0.05, a) for a in (1.0 - defect_pos_ratios - 0.1).tolist()]
    quality_gamma = 2.0
    defect_gammas = [2.0] * len(defect_columns)

    print(f"Quality pos ratio: {quality_pos_ratio:.4f} -> alpha = {quality_alpha:.4f}")
    for name, ratio, alpha in zip(DEFECT_NAMES, defect_pos_ratios, defect_alphas):
        print(f"  {name}: pos={ratio:.4f}, alpha={alpha:.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    n_folds = 5
    seeds = [42, 2024, 418, 122, 1756]
    folds = build_5fold_splits(len(data_frame), n_folds=n_folds, seeds=seeds)

    fold_stats = []
    for fold_idx, (train_idx, val_idx, seed) in enumerate(folds):
        print("\n" + "=" * 70)
        print(f"FOLD {fold_idx + 1}/{n_folds} | seed={seed} | "
              f"train={len(train_idx)} | val={len(val_idx)}")
        print("=" * 70)

        stats = train_single_fold(
            data_frame=data_frame,
            train_indices=train_idx,
            val_indices=val_idx,
            defect_alphas=defect_alphas,
            defect_gammas=defect_gammas,
            quality_alpha=quality_alpha,
            quality_gamma=quality_gamma,
            seed=seed,
            device=device,
            save_path=f'best_model_fold{fold_idx}.pth',
        )
        fold_stats.append(stats)

    print("\n" + "=" * 70)
    print("5-FOLD CV SUMMARY")
    print("=" * 70)
    for s in fold_stats:
        print(f"Fold seed={s['seed']:>4}: "
              f"F1 Q={s['best_f1_quality']:.4f}, "
              f"F1 A={s['best_f1_artefacts']:.4f}, "
              f"F1 Final={s['best_f1_final']:.4f}  ({s['save_path']})")
    print("-" * 70)
    print(f"Mean  F1 Quality  : {np.mean([s['best_f1_quality'] for s in fold_stats]):.4f}")
    print(f"Mean  F1 Artefacts: {np.mean([s['best_f1_artefacts'] for s in fold_stats]):.4f}")
    print(f"Mean  F1 Final    : {np.mean([s['best_f1_final'] for s in fold_stats]):.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()