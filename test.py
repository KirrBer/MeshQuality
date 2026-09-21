import os
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

# ==================== КОНСТАНТЫ ====================
SEED = 42
QUALITY_THRESHOLD = 0.66
DEFECT_NAMES = [
    'abstract', 'artifacts', 'intersection', 'lowpoly', 'noisy',
    'open', 'partial', 'scale', 'set', 'simple'
]
DEFAULT_DEFECT_THRESHOLDS = {
    'abstract': 0.53,
    'artifacts': 0.51,
    'intersection': 0.41,
    'lowpoly': 0.58,
    'noisy': 0.42,
    'open': 0.47,
    'partial': 0.66,
    'scale': 0.48,
    'set': 0.63,
    'simple': 0.53,
}
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
    ws = torch.initial_seed() % (2 ** 31)
    np.random.seed(ws)
    random.seed(ws)
    torch.manual_seed(ws)


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


# ==================== POINTNET (усиленный) ====================
class PointNetEncoder(nn.Module):
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


# ==================== ДАТАСЕТ ====================
class InferenceDataset(Dataset):
    def __init__(self, csv_file, data_dir, mesh_dir=None, num_points=NUM_POINTS):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.mesh_dir = mesh_dir
        self.num_points = num_points
        self.transform = self._get_transform()

    def _get_transform(self):
        return transforms.Compose([
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
                views.append(self.transform(crop))

        views = torch.stack(views)

        # Детерминированное семплирование точек (crc32 по item_id)
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


# ==================== МОДЕЛЬ ====================
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


class GatedFusion(nn.Module):
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


# ==================== TTA ====================
def _rot90(v, k):
    return torch.rot90(v, k=k, dims=[-2, -1])


def _scale(v, factor):
    B, V, C, H, W = v.shape
    new_h, new_w = int(round(H * factor)), int(round(W * factor))
    x = F.interpolate(
        v.view(B * V, C, H, W),
        size=(new_h, new_w),
        mode='bilinear', align_corners=False,
    )
    if factor >= 1.0:
        top = (new_h - H) // 2
        left = (new_w - W) // 2
        x = x[:, :, top:top + H, left:left + W]
    else:
        pad_h = (H - new_h) // 2
        pad_w = (W - new_w) // 2
        x = F.pad(x, (pad_w, W - new_w - pad_w, pad_h, H - new_h - pad_h),
                  mode='replicate')
    return x.view(B, V, C, H, W)


def _shift(v, dx, dy):
    return torch.roll(v, shifts=(dy, dx), dims=(-2, -1))


def build_tta_transforms(use_tta: bool):
    transforms_list = [lambda v: v]
    if not use_tta:
        return transforms_list
    transforms_list += [
        lambda v: torch.flip(v, dims=[-1]),
        lambda v: torch.flip(v, dims=[-2]),
        lambda v: torch.flip(v, dims=[-1, -2]),
        lambda v: _scale(v, 1.10),
        lambda v: _scale(v, 0.90),
        lambda v: _shift(v, dx=16, dy=0),
        lambda v: _shift(v, dx=0, dy=16),
        lambda v: _shift(v, dx=-16, dy=0),
    ]
    return transforms_list


# ==================== ИНФЕРЕНС ====================
def resolve_thresholds(thresholds, defect_names):
    if thresholds is None:
        return {
            'quality': QUALITY_THRESHOLD,
            'defects': {name: DEFAULT_DEFECT_THRESHOLDS.get(name, QUALITY_THRESHOLD)
                        for name in defect_names},
        }
    defects = thresholds.get('defects', {})
    if isinstance(defects, dict):
        defect_map = {name: defects.get(name, DEFAULT_DEFECT_THRESHOLDS.get(name, QUALITY_THRESHOLD))
                      for name in defect_names}
    else:
        defect_map = {name: float(thr) for name, thr in zip(defect_names, defects)}
    return {
        'quality': float(thresholds.get('quality', QUALITY_THRESHOLD)),
        'defects': defect_map,
    }


def predict(models, dataloader, device, thresholds=None,
            with_probs=False, use_tta=True):
    for m in models:
        m.eval()

    thresholds = resolve_thresholds(thresholds, DEFECT_NAMES)
    tta_transforms = build_tta_transforms(use_tta)
    n_views_total = len(tta_transforms) * len(models)
    print(f"  TTA-вариантов: {len(tta_transforms)} | моделей: {len(models)} "
          f"| усредняем по {n_views_total} предсказаниям")

    all_item_ids = []
    all_defect_probs = []
    all_quality_probs = []

    with torch.no_grad():
        for batch in dataloader:
            views, points, item_ids = batch
            views = views.to(device)
            points = points.to(device)

            defect_probs_sum = None
            quality_probs_sum = None
            n_preds = 0

            for tta in tta_transforms:
                views_aug = tta(views)
                for model in models:
                    defect_outputs, quality_output = model(views_aug, points)
                    defect_probs = torch.stack(
                        [torch.sigmoid(o) for o in defect_outputs], dim=1
                    )
                    quality_probs = torch.sigmoid(quality_output)

                    if defect_probs_sum is None:
                        defect_probs_sum = defect_probs
                        quality_probs_sum = quality_probs
                    else:
                        defect_probs_sum = defect_probs_sum + defect_probs
                        quality_probs_sum = quality_probs_sum + quality_probs
                    n_preds += 1

            defect_probs = (defect_probs_sum / n_preds).cpu().numpy()
            quality_probs = (quality_probs_sum / n_preds).cpu().numpy()

            all_item_ids.extend(item_ids)
            all_defect_probs.append(defect_probs)
            all_quality_probs.extend(quality_probs)

    all_defect_probs = np.vstack(all_defect_probs)
    all_quality_probs = np.asarray(all_quality_probs)

    defect_preds = np.stack([
        all_defect_probs[:, i] > thresholds['defects'][DEFECT_NAMES[i]]
        for i in range(len(DEFECT_NAMES))
    ], axis=1).astype(int)

    quality_preds = (all_quality_probs > thresholds['quality']).astype(int)

    results = pd.DataFrame({
        'item_id': all_item_ids,
        'abstract': defect_preds[:, 0],
        'artifacts': defect_preds[:, 1],
        'intersection': defect_preds[:, 2],
        'lowpoly': defect_preds[:, 3],
        'noisy': defect_preds[:, 4],
        'open': defect_preds[:, 5],
        'partial': defect_preds[:, 6],
        'scale': defect_preds[:, 7],
        'set': defect_preds[:, 8],
        'simple': defect_preds[:, 9],
        'quality': quality_preds,
    })

    if with_probs:
        results_probs = pd.DataFrame({
            'item_id': all_item_ids,
            'abstract': all_defect_probs[:, 0],
            'artifacts': all_defect_probs[:, 1],
            'intersection': all_defect_probs[:, 2],
            'lowpoly': all_defect_probs[:, 3],
            'noisy': all_defect_probs[:, 4],
            'open': all_defect_probs[:, 5],
            'partial': all_defect_probs[:, 6],
            'scale': all_defect_probs[:, 7],
            'set': all_defect_probs[:, 8],
            'simple': all_defect_probs[:, 9],
            'quality': all_quality_probs,
        })
        return results, results_probs

    return results


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='Инференс ансамбля моделей (visual + PointNet) с TTA')
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument(
        '--model_paths', type=str,
        default='best_model_fold0.pth,best_model_fold1.pth,best_model_fold2.pth,best_model_fold3.pth,best_model_fold4.pth',
    )
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_probs', action='store_true')
    parser.add_argument('--mesh_dir', type=str, default=None)
    parser.add_argument('--pointnet_feature_dim', type=int, default=512)
    parser.add_argument('--num_points', type=int, default=NUM_POINTS)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_tta', dest='use_tta', action='store_true')
    parser.add_argument('--no_tta', dest='use_tta', action='store_false')
    parser.set_defaults(use_tta=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    print(f'TTA: {"ON" if args.use_tta else "OFF"}')

    test_dataset = InferenceDataset(
        csv_file=args.test_csv,
        data_dir=args.test_dir,
        mesh_dir=args.mesh_dir,
        num_points=args.num_points,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    model_paths = [p.strip() for p in args.model_paths.split(',') if p.strip()]
    models = []
    for mp in model_paths:
        if not os.path.exists(mp):
            print(f'⚠ Файл не найден, пропускаю: {mp}')
            continue
        model = MeshAwareModel(
            num_views=6,
            num_defects=10,
            backbone=args.backbone,
            pointnet_feature_dim=args.pointnet_feature_dim,
        ).to(device)

        checkpoint = torch.load(mp, map_location=device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        models.append(model)
        print(f'✓ Загружена модель: {mp}')

    if len(models) == 0:
        raise RuntimeError('Ни одна модель не была загружена. Проверьте --model_paths.')

    print(f'Итого моделей в ансамбле: {len(models)}')

    thresholds = {
        'quality': QUALITY_THRESHOLD,
        'defects': DEFAULT_DEFECT_THRESHOLDS,
    }

    print('\nПороги:')
    print(f'  Quality: {thresholds["quality"]}')
    for name in DEFECT_NAMES:
        print(f'  {name}: {thresholds["defects"][name]}')

    print('\nНачало инференса...')
    if args.save_probs:
        results, results_probs = predict(
            models, test_loader, device,
            thresholds=thresholds, with_probs=True, use_tta=args.use_tta,
        )
    else:
        results = predict(
            models, test_loader, device,
            thresholds=thresholds, with_probs=False, use_tta=args.use_tta,
        )

    results.to_csv(args.output, index=False)
    print(f'\n✅ Результаты сохранены в {args.output}')

    print('\n' + '=' * 50)
    print('СТАТИСТИКА ПРЕДСКАЗАНИЙ')
    print('=' * 50)
    print(f'Всего образцов: {len(results)}')
    print(f'Quality = 1: {results["quality"].sum()} ({results["quality"].mean() * 100:.1f}%)')
    print(f'Quality = 0: {len(results) - results["quality"].sum()} ({(1 - results["quality"].mean()) * 100:.1f}%)')

    defect_cols = [col for col in results.columns if col not in ('quality', 'item_id')]
    for col in defect_cols:
        count = results[col].sum()
        print(f'{col}: {count} ({count / len(results) * 100:.1f}%)')
    print('=' * 50)

    if args.save_probs:
        probs_output = args.output.replace('.csv', '_with_probs.csv')
        results_probs.to_csv(probs_output, index=False)
        print(f'✅ Результаты с вероятностями сохранены в {probs_output}')


if __name__ == '__main__':
    main()