import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0, densenet121, vit_b_16
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import os
from PIL import Image
import time
from tqdm import tqdm
import warnings
import gc
import logging
from pathlib import Path
import platform
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json

# YOLO için ek importlar
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ Ultralytics YOLO bulunamadı. Pip install ultralytics ile kurun.")
    YOLO_AVAILABLE = False

warnings.filterwarnings('ignore')

# Windows-specific configuration
IS_WINDOWS = platform.system() == 'Windows'

# Enhanced Configuration
CONFIG = {
    'EPOCHS': 20,
    'BATCH_SIZE_GPU': 24,
    'BATCH_SIZE_CPU': 6,
    'NUM_WORKERS': 0 if IS_WINDOWS else 4,
    'PATIENCE': 7,
    'PRINT_FREQ': 50,
    'LR': 0.0005,
    'WEIGHT_DECAY': 1e-5,
    'IMG_SIZE': 256,
    'PREFETCH_FACTOR': None,
    'PERSISTENT_WORKERS': False,
    'FOCAL_ALPHA': 0.25,
    'FOCAL_GAMMA': 2.0,
    'MIXUP_ALPHA': 0.2,
    'CUTMIX_ALPHA': 1.0,
    'LABEL_SMOOTHING': 0.1,
    'TTA_TRANSFORMS': 8,
    'GRAD_ACCUMULATION': 2,
    # YOLO Configuration
    'YOLO_CONF': 0.5,  # YOLO güven eşiği
    'YOLO_IOU': 0.45,  # YOLO IoU eşiği
    'YOLO_IMGSZ': 640,  # YOLO input size
    'MIN_DETECTION_SIZE': 50,  # Minimum tespit boyutu
    'MAX_DETECTIONS': 5,  # Maksimum tespit sayısı
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_device():
    """Enhanced device setup with better memory management"""
    logger.info("=" * 60)
    logger.info("ENHANCED PYTORCH CUDA GPU KONTROLÜ")
    logger.info("=" * 60)

    logger.info(f"Platform: {platform.system()}")
    logger.info(f"PyTorch sürümü: {torch.__version__}")
    logger.info(f"CUDA mevcut: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA sürümü: {torch.version.cuda}")
        logger.info(f"cuDNN sürümü: {torch.backends.cudnn.version()}")
        logger.info(f"GPU sayısı: {torch.cuda.device_count()}")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"GPU 0: {gpu_name} ({gpu_memory:.1f} GB)")

        device = torch.device("cuda")

        # Enhanced CUDA optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False

        # Enable mixed precision optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Memory-based batch size adjustment
        if gpu_memory <= 6:
            batch_size = 16
        elif gpu_memory <= 8:
            batch_size = 20
        elif gpu_memory <= 12:
            batch_size = 24
        else:
            batch_size = CONFIG['BATCH_SIZE_GPU']

    else:
        device = torch.device("cpu")
        batch_size = CONFIG['BATCH_SIZE_CPU']
        logger.info("GPU bulunamadı, CPU kullanılacak")

    logger.info("=" * 60)
    return device, batch_size


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MixupCutmix:
    """Enhanced data augmentation with Mixup and Cutmix"""

    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def mixup_data(self, x, y):
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix_data(self, x, y):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y_a, y_b, lam

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, x, y):
        if np.random.rand() < self.prob:
            if np.random.rand() < 0.5:
                return self.mixup_data(x, y)
            else:
                return self.cutmix_data(x, y)
        return x, y, y, 1.0


class EnhancedBrainTumorDataset(Dataset):
    """Enhanced dataset with better augmentation and class balance handling"""

    def __init__(self, root_dir, transform=None, class_names=None, use_class_weights=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names or ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.use_class_weights = use_class_weights

        # Collect images and labels
        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                class_images = []
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        class_images.append(str(img_path))

                self.images.extend(class_images)
                self.labels.extend([class_idx] * len(class_images))

        # Class distribution analysis
        self.class_counts = Counter(self.labels)
        logger.info(f"Dataset loaded: {len(self.images)} images, {len(set(self.labels))} classes")

        for class_idx, count in self.class_counts.items():
            logger.info(f"  {self.class_names[class_idx]}: {count} images")

        # Compute class weights for balanced sampling
        if use_class_weights:
            self.class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.labels),
                y=self.labels
            )
            logger.info(f"Class weights: {dict(zip(self.class_names, self.class_weights))}")
        else:
            self.class_weights = None

    def get_sampler(self):
        """Get weighted sampler for balanced training"""
        if self.class_weights is not None:
            sample_weights = [self.class_weights[label] for label in self.labels]
            return WeightedRandomSampler(sample_weights, len(sample_weights))
        return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                if hasattr(self.transform, '__call__'):
                    # Albumentations transform
                    transformed = self.transform(image=image)
                    image = transformed['image']
                else:
                    # Torchvision transform
                    image = Image.fromarray(image)
                    image = self.transform(image)

            return image, label

        except Exception as e:
            logger.warning(f"Error loading image: {img_path} - {e}")
            # Return dummy image
            dummy_image = torch.zeros(3, CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])
            return dummy_image, label


class AttentionBlock(nn.Module):
    """Spatial Attention Block for enhanced feature learning"""

    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class EnhancedCNNModel(nn.Module):
    """Enhanced CNN with attention and better architecture"""

    def __init__(self, num_classes=4, dropout=0.3):
        super(EnhancedCNNModel, self).__init__()

        # Enhanced feature extraction with attention
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            AttentionBlock(64),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            AttentionBlock(128),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            AttentionBlock(256),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Enhanced classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class BrainTumorClassifier:
    """
    Eğitilmiş modelleri yükleyip tahmin yapan sınıf
    """

    def __init__(self, model_dir='enhanced_pytorch_models', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.img_size = CONFIG['IMG_SIZE']
        self.models = {}
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        # Transforms
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # TTA transforms
        self.tta_transforms = self.get_tta_transforms()

        # Modelleri yükle
        self.load_trained_models(model_dir)

    def get_tta_transforms(self):
        """TTA için transforms"""
        tta_list = []

        # Original
        tta_list.append(A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))

        # Flip variations
        tta_list.append(A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))

        tta_list.append(A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))

        # Rotation
        tta_list.append(A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Rotate(limit=15, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))

        return tta_list

    def load_trained_models(self, model_dir):
        """Eğitilmiş modelleri yükle"""
        model_dir = Path(model_dir)
        if not model_dir.exists():
            logger.warning(f"Model dizini bulunamadı: {model_dir}")
            return

        # Model dosyalarını bul
        model_files = list(model_dir.glob("best_*_enhanced.pth"))

        for model_file in model_files:
            try:
                model_name = model_file.stem.replace("best_", "").replace("_enhanced", "")

                checkpoint = torch.load(model_file, map_location=self.device)

                # Model tipine göre yükle
                if model_name == 'enhanced_cnn':
                    model = EnhancedCNNModel(num_classes=4).to(self.device)
                elif model_name == 'efficientnet_b0':
                    model = efficientnet_b0(weights=None)
                    model.classifier = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(model.classifier[1].in_features, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Linear(512, 4)
                    )
                    model = model.to(self.device)
                elif model_name == 'densenet121':
                    model = densenet121(weights=None)
                    model.classifier = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(model.classifier.in_features, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Linear(512, 4)
                    )
                    model = model.to(self.device)
                elif model_name == 'resnet50_enhanced':
                    model = resnet50(weights=None)
                    model.fc = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(model.fc.in_features, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 4)
                    )
                    model = model.to(self.device)
                else:
                    continue

                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models[model_name] = model

                logger.info(f"✅ {model_name} yüklendi")

            except Exception as e:
                logger.error(f"❌ {model_file} yüklenemedi: {e}")
                continue

        logger.info(f"Toplam {len(self.models)} model yüklendi")

    def predict_single_image(self, image, use_tta=True):
        """Tek görüntü için tahmin"""
        if len(self.models) == 0:
            logger.error("Hiçbir model yüklenmedi!")
            return None

        try:
            # Görüntü preprocessing
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            predictions = {}

            # Her model için tahmin
            for name, model in self.models.items():
                model.eval()
                model_preds = []

                with torch.no_grad():
                    if use_tta:
                        # TTA ile tahmin
                        for transform in self.tta_transforms:
                            transformed = transform(image=image)
                            input_tensor = transformed['image'].unsqueeze(0).to(self.device)

                            if self.scaler:
                                with torch.cuda.amp.autocast():
                                    output = model(input_tensor)
                            else:
                                output = model(input_tensor)

                            probs = torch.softmax(output, dim=1)
                            model_preds.append(probs.cpu().numpy())

                        # TTA sonuçlarını ortala
                        final_pred = np.mean(model_preds, axis=0)[0]
                    else:
                        # Tek tahmin
                        transformed = self.transform(image=image)
                        input_tensor = transformed['image'].unsqueeze(0).to(self.device)

                        if self.scaler:
                            with torch.cuda.amp.autocast():
                                output = model(input_tensor)
                        else:
                            output = model(input_tensor)

                        final_pred = torch.softmax(output, dim=1).cpu().numpy()[0]

                    predictions[name] = final_pred

            # Ensemble prediction
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            pred_class = np.argmax(ensemble_pred)
            confidence = ensemble_pred[pred_class]

            return {
                'class': self.class_names[pred_class],
                'confidence': float(confidence),
                'class_index': int(pred_class),
                'probabilities': {name: float(prob) for name, prob in zip(self.class_names, ensemble_pred)},
                'individual_predictions': {name: pred.tolist() for name, pred in predictions.items()}
            }

        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return None


class YOLOBrainDetector:
    """
    YOLO ile beyin tümörü bölgesi tespiti - İyileştirilmiş versiyon
    """

    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.yolo_model = None

        if YOLO_AVAILABLE:
            self.load_yolo_model(model_path)
        else:
            logger.warning("YOLO mevcut değil, tüm görüntü kullanılacak")

    def load_yolo_model(self, model_path=None):
        """YOLO modelini yükle"""
        try:
            if model_path and Path(model_path).exists():
                # Özel beyin tümörü YOLO modeli
                self.yolo_model = YOLO(model_path)
                logger.info(f"Özel YOLO modeli yüklendi: {model_path}")
            else:
                # Genel YOLO modeli kullan ama daha hassas ayarlarla
                self.yolo_model = YOLO('yolov8n.pt')  # veya yolov8s.pt daha iyi sonuçlar için
                logger.info("Genel YOLO modeli yüklendi")

        except Exception as e:
            logger.error(f"YOLO model yükleme hatası: {e}")
            self.yolo_model = None

    def detect_brain_regions(self, image, conf_threshold=None):
        """Görüntüde beyin bölgelerini tespit et - İyileştirilmiş"""
        if self.yolo_model is None:
            # YOLO yoksa tüm görüntüyü kullan
            h, w = image.shape[:2]
            return [{
                'bbox': [0, 0, w, h],
                'confidence': 1.0,
                'class': 'brain',
                'area': w * h
            }]

        try:
            # Daha hassas parametreler
            conf = conf_threshold or 0.25  # Düşük confidence threshold
            iou_threshold = 0.3  # Düşük IoU threshold

            # YOLO inference
            results = self.yolo_model(
                image,
                conf=conf,
                iou=iou_threshold,
                imgsz=640,  # Yüksek çözünürlük
                verbose=False,
                agnostic_nms=True,  # Class-agnostic NMS
                max_det=10  # Daha fazla detection
            )

            detections = []
            h, w = image.shape[:2]

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()

                        # Koordinatları kontrol et ve sınırla
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(x2)), min(h, int(y2))

                        # Minimum boyut kontrolü - daha küçük değer
                        box_w, box_h = x2 - x1, y2 - y1
                        area = box_w * box_h

                        # Daha esnek boyut kontrolü
                        if area >= 20 ** 2 and box_w >= 20 and box_h >= 20:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'class': 'brain_region',
                                'area': int(area),
                                'width': box_w,
                                'height': box_h
                            })

            # Post-processing: Overlapping box'ları filtrele
            detections = self.filter_overlapping_boxes(detections)

            # Güven skoruna göre sırala
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

            # Maksimum detection sayısını artır
            detections = detections[:15]  # Daha fazla detection

            # Eğer hiç tespit yoksa, görüntüyü grid'e böl
            if not detections:
                detections = self.create_grid_detections(image)

            return detections

        except Exception as e:
            logger.error(f"YOLO tespit hatası: {e}")
            return self.create_grid_detections(image)

    def filter_overlapping_boxes(self, detections, iou_threshold=0.5):
        """Çakışan box'ları filtrele"""
        if len(detections) <= 1:
            return detections

        # IoU hesaplama
        def calculate_iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1['bbox']
            x1_2, y1_2, x2_2, y2_2 = box2['bbox']

            # Kesişim alanı
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)

            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0

            intersection = (xi2 - xi1) * (yi2 - yi1)
            union = box1['area'] + box2['area'] - intersection

            return intersection / union if union > 0 else 0.0

        # Güven skoruna göre sırala
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        filtered = []
        for det in detections:
            keep = True
            for kept_det in filtered:
                if calculate_iou(det, kept_det) > iou_threshold:
                    keep = False
                    break
            if keep:
                filtered.append(det)

        return filtered

    def create_grid_detections(self, image, grid_size=3):
        """Görüntüyü grid'e bölerek detection'lar oluştur"""
        h, w = image.shape[:2]
        detections = []

        cell_h = h // grid_size
        cell_w = w // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                x1 = j * cell_w
                y1 = i * cell_h
                x2 = min((j + 1) * cell_w, w)
                y2 = min((i + 1) * cell_h, h)

                # Merkezi bölgelere daha yüksek öncelik ver
                center_bonus = 0.3 if (i == 1 and j == 1) else 0.0
                base_conf = 0.5 + center_bonus

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': base_conf,
                    'class': f'grid_cell_{i}_{j}',
                    'area': (x2 - x1) * (y2 - y1)
                })

        return detections


class EnhancedBrainTumorSystem:
    """
    YOLO + Sınıflandırma entegre sistemi - İyileştirilmiş
    """

    def __init__(self, yolo_model_path=None, classifier_model_dir='enhanced_pytorch_models', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # YOLO detector - daha hassas ayarlar
        self.detector = YOLOBrainDetector(yolo_model_path, device)

        # Classifier
        self.classifier = BrainTumorClassifier(classifier_model_dir, device)

        # Stats
        self.stats = {
            'processed_images': 0,
            'total_detections': 0,
            'classifications': Counter()
        }

        logger.info(f"Enhanced Brain Tumor System başlatıldı - Device: {self.device}")

    def extract_roi_smart(self, image, bbox, padding=0.05):
        """Akıllı ROI çıkarma - daha küçük padding"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Dinamik padding - küçük box'lar için daha az padding
        box_w, box_h = x2 - x1, y2 - y1
        avg_size = (box_w + box_h) / 2

        # Küçük box'lar için padding azalt
        if avg_size < 100:
            padding = 0.02
        elif avg_size < 200:
            padding = 0.03
        else:
            padding = 0.05

        # Padding ekle
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        roi = image[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)

    def draw_results_on_image(self, image, results):
        """Sonuçları görüntü üzerine çiz - İyileştirilmiş görsel"""
        if not results or not results['results']:
            return image

        # Farklı renkler tanımla
        colors = [
            (0, 255, 0),  # Yeşil - glioma
            (255, 0, 0),  # Mavi - meningioma
            (0, 255, 255),  # Sarı - notumor
            (255, 0, 255),  # Magenta - pituitary
            (255, 255, 0),  # Cyan - diğer
        ]

        class_color_map = {
            'glioma': (0, 255, 0),
            'meningioma': (255, 0, 0),
            'notumor': (0, 255, 255),
            'pituitary': (255, 0, 255)
        }

        for result in results['results']:
            detection = result['detection']
            classification = result['classification']
            bbox = result['adjusted_bbox']

            x1, y1, x2, y2 = bbox

            # Sınıfa göre renk seç
            color = class_color_map.get(classification['class'], (255, 255, 255))

            # Güven skoruna göre çizgi kalınlığı
            thickness = 3 if result['combined_confidence'] > 0.7 else 2

            # Daha ince bounding box çiz
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Kompakt label
            conf_text = f"{classification['confidence']:.2f}"
            main_label = f"{classification['class']}: {conf_text}"

            # Label arka planı - daha küçük
            label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 6),
                          (x1 + label_size[0] + 4, y1), color, -1)

            # Ana label yazısı - daha küçük font
            cv2.putText(image, main_label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Sadece yüksek probability'leri göster
            y_offset = y2 + 15
            for class_name, prob in classification['probabilities'].items():
                if prob > 0.15 and class_name != classification['class']:  # Sadece anlamlı ve ana sınıf olmayan
                    prob_text = f"{class_name}: {prob:.2f}"
                    cv2.putText(image, prob_text, (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                    y_offset += 12

        return image

    def process_single_image(self, image_input, save_results=False, output_dir='results'):
        """Tek görüntü işle - İyileştirilmiş"""
        try:
            # Görüntüyü yükle
            if isinstance(image_input, (str, Path)):
                image_path = str(image_input)
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Görüntü yüklenemedi: {image_path}")
                    return None
                original_image = image.copy()
            else:
                image = image_input.copy()
                original_image = image.copy()
                image_path = "unknown"

            self.stats['processed_images'] += 1

            # 1. YOLO ile beyin bölgelerini tespit et - daha hassas
            detections = self.detector.detect_brain_regions(image, conf_threshold=0.2)
            self.stats['total_detections'] += len(detections)

            # Detections'ları filtrele ve optimize et
            detections = self.optimize_detections(detections, image.shape)

            results = []

            # 2. Her tespit için sınıflandırma yap
            for i, detection in enumerate(detections):
                bbox = detection['bbox']

                # Akıllı ROI çıkar
                roi, adjusted_bbox = self.extract_roi_smart(image, bbox)

                if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                    continue

                # Sınıflandırma yap
                classification = self.classifier.predict_single_image(roi, use_tta=False)  # TTA kapalı, hız için

                if classification:
                    self.stats['classifications'][classification['class']] += 1

                    # Kalite skorları ekle
                    quality_score = self.calculate_detection_quality(detection, classification, roi)

                    result = {
                        'detection_id': i,
                        'detection': detection,
                        'adjusted_bbox': adjusted_bbox,
                        'classification': classification,
                        'roi_shape': roi.shape,
                        'combined_confidence': detection['confidence'] * classification['confidence'],
                        'quality_score': quality_score
                    }
                    results.append(result)

            # Sonuçları kalite skoruna göre filtrele
            results = [r for r in results if r['quality_score'] > 0.3]
            results = sorted(results, key=lambda x: x['quality_score'], reverse=True)

            final_result = {
                'image_path': image_path,
                'image_shape': original_image.shape,
                'num_detections': len(results),
                'results': results,
                'processing_stats': {
                    'yolo_detections': len(detections),
                    'successful_classifications': len(results)
                }
            }

            # Sonuçları kaydet
            if save_results:
                self.save_result(original_image, final_result, output_dir)

            return final_result

        except Exception as e:
            logger.error(f"Görüntü işleme hatası: {e}")
            return None

    def optimize_detections(self, detections, image_shape):
        """Detection'ları optimize et"""
        h, w = image_shape[:2]

        # Çok küçük detection'ları filtrele
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            box_w, box_h = x2 - x1, y2 - y1

            # Minimum boyut kontrolü
            if box_w >= 15 and box_h >= 15:
                # Görüntü sınırlarını kontrol et
                if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                    filtered.append(det)

        return filtered

    def calculate_detection_quality(self, detection, classification, roi):
        """Detection kalitesini hesapla"""
        # Temel skorlar
        det_conf = detection['confidence']
        class_conf = classification['confidence']

        # ROI boyutu skoru
        roi_area = roi.shape[0] * roi.shape[1]
        size_score = min(1.0, roi_area / (100 * 100))  # 100x100'e normalize

        # Aspect ratio skoru
        aspect_ratio = roi.shape[1] / roi.shape[0]
        aspect_score = 1.0 - abs(1.0 - aspect_ratio) * 0.5  # Kare'ye yakınlık

        # Final kalite skoru
        quality_score = (det_conf * 0.3 + class_conf * 0.4 +
                         size_score * 0.2 + aspect_score * 0.1)

        return quality_score


class EnhancedBrainTumorSystem:
    """
    YOLO + Sınıflandırma entegre sistemi
    """

    def __init__(self, yolo_model_path=None, classifier_model_dir='enhanced_pytorch_models', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # YOLO detector
        self.detector = YOLOBrainDetector(yolo_model_path, device)

        # Classifier
        self.classifier = BrainTumorClassifier(classifier_model_dir, device)

        # Stats
        self.stats = {
            'processed_images': 0,
            'total_detections': 0,
            'classifications': Counter()
        }

        logger.info(f"Enhanced Brain Tumor System başlatıldı - Device: {self.device}")

    def extract_roi(self, image, bbox, padding=0.1):
        """ROI çıkar ve padding ekle"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Padding ekle
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        roi = image[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)

    def process_single_image(self, image_input, save_results=False, output_dir='results'):
        """Tek görüntü işle"""
        try:
            # Görüntüyü yükle
            if isinstance(image_input, (str, Path)):
                image_path = str(image_input)
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Görüntü yüklenemedi: {image_path}")
                    return None
                original_image = image.copy()
            else:
                image = image_input.copy()
                original_image = image.copy()
                image_path = "unknown"

            self.stats['processed_images'] += 1

            # 1. YOLO ile beyin bölgelerini tespit et
            detections = self.detector.detect_brain_regions(image)
            self.stats['total_detections'] += len(detections)

            results = []

            # 2. Her tespit için sınıflandırma yap
            for i, detection in enumerate(detections):
                bbox = detection['bbox']

                # ROI çıkar
                roi, adjusted_bbox = self.extract_roi(image, bbox)

                if roi.size == 0:
                    continue

                # Sınıflandırma yap
                classification = self.classifier.predict_single_image(roi, use_tta=True)

                if classification:
                    self.stats['classifications'][classification['class']] += 1

                    result = {
                        'detection_id': i,
                        'detection': detection,
                        'adjusted_bbox': adjusted_bbox,
                        'classification': classification,
                        'roi_shape': roi.shape,
                        'combined_confidence': detection['confidence'] * classification['confidence']
                    }
                    results.append(result)

            final_result = {
                'image_path': image_path,
                'image_shape': original_image.shape,
                'num_detections': len(results),
                'results': results,
                'processing_stats': {
                    'yolo_detections': len(detections),
                    'successful_classifications': len(results)
                }
            }

            # Sonuçları kaydet
            if save_results:
                self.save_result(original_image, final_result, output_dir)

            return final_result

        except Exception as e:
            logger.error(f"Görüntü işleme hatası: {e}")
            return None

    def process_video(self, video_source, output_path=None, display=True, save_frames=False):
        """Video işle (canlı sistem için)"""
        # Video kaynağını aç
        if video_source == 'webcam' or video_source == 0:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            logger.error(f"Video kaynağı açılamadı: {video_source}")
            return

        # Video writer setup
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        fps_counter = []

        logger.info("Video işleme başladı. ESC tuşu ile çıkış yapabilirsiniz.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                start_time = time.time()

                # Her N frame'i işle (performans için)
                if frame_count % 3 == 0:  # Her 3. frame'i işle
                    result = self.process_single_image(frame)

                    if result and result['results']:
                        frame = self.draw_results_on_image(frame, result)

                # FPS hesapla
                processing_time = time.time() - start_time
                if processing_time > 0:
                    current_fps = 1.0 / processing_time
                    fps_counter.append(current_fps)
                    if len(fps_counter) > 30:
                        fps_counter.pop(0)
                    avg_fps = sum(fps_counter) / len(fps_counter)
                else:
                    avg_fps = 0

                # FPS ve stats göster
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Processed: {self.stats['processed_images']}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Görüntüyü göster
                if display:
                    cv2.imshow('Enhanced Brain Tumor Detection', frame)

                # Video kaydet
                if writer:
                    writer.write(frame)

                # Frame kaydet
                if save_frames and frame_count % 30 == 0:  # Her 30 frame'de bir kaydet
                    cv2.imwrite(f'frame_{frame_count:06d}.jpg', frame)

                # ESC tuşu kontrolü
                if display and cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            logger.info("Kullanıcı tarafından durduruldu")
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            logger.info(f"Video işleme tamamlandı. İşlenen frame: {frame_count}")
            logger.info(f"İstatistikler: {self.stats}")

    def draw_results_on_image(self, image, results):
        """Sonuçları görüntü üzerine çiz"""
        if not results or not results['results']:
            return image

        for result in results['results']:
            detection = result['detection']
            classification = result['classification']
            bbox = result['adjusted_bbox']

            x1, y1, x2, y2 = bbox

            # Güven skoruna göre renk belirle
            combined_conf = result['combined_confidence']
            if combined_conf > 0.8:
                color = (0, 255, 0)  # Yeşil - yüksek güven
            elif combined_conf > 0.6:
                color = (0, 255, 255)  # Sarı - orta güven
            else:
                color = (0, 0, 255)  # Kırmızı - düşük güven

            # Bounding box çiz
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Ana label
            main_label = f"{classification['class']}: {classification['confidence']:.3f}"

            # Label arka plan
            label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)

            # Ana label yazısı
            cv2.putText(image, main_label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Detaylı probabilities
            y_offset = y1 + 20
            for class_name, prob in classification['probabilities'].items():
                if prob > 0.05:  # Sadece anlamlı probabilities
                    prob_text = f"{class_name}: {prob:.3f}"
                    cv2.putText(image, prob_text, (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 15

            # Detection confidence
            det_text = f"Det: {detection['confidence']:.3f}"
            cv2.putText(image, det_text, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return image

    def save_result(self, image, result, output_dir):
        """Sonuçları kaydet"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())

            # Görüntüyü kaydet
            result_image = self.draw_results_on_image(image.copy(), result)
            cv2.imwrite(str(output_dir / f"result_{timestamp}.jpg"), result_image)

            # JSON sonucu kaydet
            json_result = {
                'timestamp': timestamp,
                'image_shape': result['image_shape'],
                'num_detections': result['num_detections'],
                'results': []
            }

            for res in result['results']:
                json_result['results'].append({
                    'detection_id': res['detection_id'],
                    'bbox': res['adjusted_bbox'],
                    'detection_confidence': res['detection']['confidence'],
                    'classification': res['classification']['class'],
                    'classification_confidence': res['classification']['confidence'],
                    'combined_confidence': res['combined_confidence'],
                    'probabilities': res['classification']['probabilities']
                })

            with open(output_dir / f"result_{timestamp}.json", 'w') as f:
                json.dump(json_result, f, indent=2)

        except Exception as e:
            logger.error(f"Sonuç kaydetme hatası: {e}")

    def batch_process(self, input_dir, output_dir='batch_results'):
        """Toplu işlem"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Görüntü dosyalarını bul
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        logger.info(f"Toplam {len(image_files)} görüntü bulundu")

        results_summary = []

        for i, image_file in enumerate(image_files):
            logger.info(f"İşleniyor ({i + 1}/{len(image_files)}): {image_file.name}")

            result = self.process_single_image(image_file, save_results=True, output_dir=output_dir)

            if result:
                summary = {
                    'file': image_file.name,
                    'detections': result['num_detections'],
                    'classifications': [r['classification']['class'] for r in result['results']]
                }
                results_summary.append(summary)

        # Özet raporu kaydet
        with open(output_dir / 'batch_summary.json', 'w') as f:
            json.dump({
                'total_images': len(image_files),
                'processed_successfully': len(results_summary),
                'overall_stats': dict(self.stats),
                'results': results_summary
            }, f, indent=2)

        logger.info(f"Toplu işlem tamamlandı. Sonuçlar: {output_dir}")


class EnhancedBrainTumorEnsemble:
    """Enhanced ensemble with better training strategies"""

    def __init__(self, device, batch_size, num_classes=4, img_size=256):
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = device
        self.batch_size = batch_size
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.models = {}
        self.histories = {}
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        self.mixup_cutmix = MixupCutmix(CONFIG['MIXUP_ALPHA'], CONFIG['CUTMIX_ALPHA'])

        logger.info(f"Enhanced Ensemble System - Device: {self.device}, Batch Size: {self.batch_size}")

    def get_albumentations_transforms(self):
        """Enhanced Albumentations transforms"""
        train_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.OneOf([
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
            ], p=0.8),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.3),
                A.ElasticTransform(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
            ], p=0.5),
            A.OneOf([
                A.HueSaturationValue(p=0.3),
                A.RGBShift(p=0.3),
                A.ChannelShuffle(p=0.1),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        val_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        return train_transform, val_transform

    def get_tta_transforms(self):
        """Test Time Augmentation transforms"""
        tta_transforms = []
        base_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        tta_transforms.append(base_transform)  # Original

        # Add variations
        variations = [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Rotate(limit=15, p=1.0),
            A.Rotate(limit=-15, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.CLAHE(clip_limit=2, p=1.0),
        ]

        for var in variations:
            tta_transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                var,
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            tta_transforms.append(tta_transform)

        return tta_transforms[:CONFIG['TTA_TRANSFORMS']]

    def prepare_enhanced_data(self, train_dir, test_dir):
        """Enhanced data preparation with class balancing"""
        logger.info("Enhanced veri yükleyicileri hazırlanıyor...")

        train_transform, val_transform = self.get_albumentations_transforms()

        # Create datasets with class weighting
        full_train_dataset = EnhancedBrainTumorDataset(
            train_dir, train_transform, self.class_names, use_class_weights=True
        )
        test_dataset = EnhancedBrainTumorDataset(
            test_dir, val_transform, self.class_names
        )

        # Enhanced train/validation split with stratification
        from sklearn.model_selection import train_test_split

        train_indices, val_indices = train_test_split(
            range(len(full_train_dataset)),
            test_size=0.15,
            stratify=full_train_dataset.labels,
            random_state=42
        )

        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)

        # Change validation transform
        val_dataset.dataset.transform = val_transform

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        # Enhanced data loaders with weighted sampling
        train_sampler = full_train_dataset.get_sampler()
        if train_sampler:
            # Use subset of sampler for training data
            train_weights = [full_train_dataset.class_weights[full_train_dataset.labels[i]]
                             for i in train_indices]
            train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        common_params = {
            'num_workers': CONFIG['NUM_WORKERS'],
            'pin_memory': True if self.device.type == 'cuda' else False,
            'prefetch_factor': CONFIG['PREFETCH_FACTOR'],
            'persistent_workers': CONFIG['PERSISTENT_WORKERS'],
        }

        if CONFIG['NUM_WORKERS'] == 0:
            common_params.pop('prefetch_factor', None)
            common_params.pop('persistent_workers', None)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size // CONFIG['GRAD_ACCUMULATION'],
            sampler=train_sampler,
            drop_last=True,
            **common_params
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **common_params
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **common_params
        )

        return train_loader, val_loader, test_loader

    def create_enhanced_models(self):
        """Create enhanced model ensemble"""
        logger.info("Enhanced ensemble modelleri oluşturuluyor...")

        models = {}

        # 1. Enhanced Custom CNN with Attention
        try:
            models['enhanced_cnn'] = EnhancedCNNModel(self.num_classes).to(self.device)
            logger.info("✅ Enhanced CNN with Attention oluşturuldu")
        except Exception as e:
            logger.error(f"❌ Enhanced CNN hatası: {e}")

        # 2. EfficientNet-B0 (Memory efficient)
        try:
            efficient = efficientnet_b0(weights='IMAGENET1K_V1')
            # Freeze early layers
            for param in list(efficient.parameters())[:-10]:
                param.requires_grad = False

            efficient.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(efficient.classifier[1].in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_classes)
            )
            models['efficientnet_b0'] = efficient.to(self.device)
            logger.info("✅ EfficientNet-B0 oluşturuldu")
        except Exception as e:
            logger.error(f"❌ EfficientNet-B0 hatası: {e}")

        # 3. DenseNet121 (if GPU memory allows)
        if self.device.type == 'cuda':
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                if gpu_memory >= 8:  # Only on GPUs with >=8GB
                    densenet = densenet121(weights='IMAGENET1K_V1')
                    # Freeze early layers
                    for param in list(densenet.parameters())[:-15]:
                        param.requires_grad = False

                    densenet.classifier = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(densenet.classifier.in_features, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Linear(512, self.num_classes)
                    )
                    models['densenet121'] = densenet.to(self.device)
                    logger.info("✅ DenseNet121 oluşturuldu")
            except Exception as e:
                logger.error(f"❌ DenseNet121 hatası: {e}")

        # 4. ResNet50 with enhanced head
        if self.device.type == 'cuda':
            try:
                resnet = resnet50(weights='IMAGENET1K_V2')
                # Adaptive freezing
                for param in list(resnet.parameters())[:-8]:
                    param.requires_grad = False

                resnet.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(resnet.fc.in_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, self.num_classes)
                )
                models['resnet50_enhanced'] = resnet.to(self.device)
                logger.info("✅ Enhanced ResNet50 oluşturuldu")
            except Exception as e:
                logger.error(f"❌ Enhanced ResNet50 hatası: {e}")

        self.models = models
        logger.info(f"Toplam {len(models)} model oluşturuldu: {list(models.keys())}")

        # Display model parameters
        total_params = 0
        for name, model in models.items():
            params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params += trainable_params
            logger.info(f"{name}: {params:,} total, {trainable_params:,} trainable")

        logger.info(f"Toplam eğitilebilir parametreler: {total_params:,}")

    def enhanced_train_model(self, model, train_loader, val_loader, epochs=20, model_name="model"):
        """Enhanced training with advanced techniques"""
        logger.info(f"🚀 {model_name} enhanced eğitimi başlıyor...")

        # Memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        # Enhanced loss function
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['LABEL_SMOOTHING'])

        # Enhanced optimizer with scheduling
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CONFIG['LR'],
            weight_decay=CONFIG['WEIGHT_DECAY'],
            betas=(0.9, 0.999)
        )

        # Enhanced scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CONFIG['LR'] * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Training history
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Training with enhanced techniques
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            optimizer.zero_grad()

            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    # Apply Mixup/Cutmix
                    mixed_data, target_a, target_b, lam = self.mixup_cutmix(data, target)

                    # Forward pass with mixed precision
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            output = model(mixed_data)
                            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)

                        # Gradient accumulation
                        loss = loss / CONFIG['GRAD_ACCUMULATION']
                        self.scaler.scale(loss).backward()

                        if (batch_idx + 1) % CONFIG['GRAD_ACCUMULATION'] == 0:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            optimizer.zero_grad()
                    else:
                        output = model(mixed_data)
                        loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)

                        loss = loss / CONFIG['GRAD_ACCUMULATION']
                        loss.backward()

                        if (batch_idx + 1) % CONFIG['GRAD_ACCUMULATION'] == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                    scheduler.step()

                    train_loss += loss.item() * CONFIG['GRAD_ACCUMULATION']
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()

                    if batch_idx % CONFIG['PRINT_FREQ'] == 0:
                        logger.info(f"Batch {batch_idx}/{len(train_loader)} - "
                                    f"Loss: {loss.item():.4f} - "
                                    f"LR: {scheduler.get_last_lr()[0]:.6f}")

                except Exception as e:
                    logger.warning(f"Training batch {batch_idx} hatası: {e}")
                    continue

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    try:
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)

                        if self.scaler:
                            with torch.cuda.amp.autocast():
                                output = model(data)
                                loss = criterion(output, target)
                        else:
                            output = model(data)
                            loss = criterion(output, target)

                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()

                    except Exception as e:
                        logger.warning(f"Validation batch hatası: {e}")
                        continue

            # Calculate metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100. * train_correct / train_total
            epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            epoch_val_acc = 100. * val_correct / val_total if val_total > 0 else 0

            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)

            # Enhanced model saving
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                patience_counter = 0
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_val_acc,
                        'config': CONFIG,
                        'class_names': self.class_names
                    }, f'best_{model_name}_enhanced.pth')
                except Exception as e:
                    logger.error(f"Model kaydetme hatası: {e}")
            else:
                patience_counter += 1

            epoch_time = time.time() - start_time
            logger.info(
                f"{model_name} Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) - "
                f"Train: {epoch_train_acc:.2f}% - Val: {epoch_val_acc:.2f}% - "
                f"Best: {best_val_acc:.2f}%"
            )

            # Enhanced early stopping
            if patience_counter >= CONFIG['PATIENCE']:
                logger.info(f"Early stopping! En iyi val acc: {best_val_acc:.2f}%")
                break

            # Memory cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        # Load best model
        try:
            checkpoint = torch.load(f'best_{model_name}_enhanced.pth', map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ {model_name} tamamlandı! En iyi Val Acc: {best_val_acc:.2f}%")
        except Exception as e:
            logger.warning(f"En iyi model yükleme hatası: {e}")

        return history

    def train_enhanced_ensemble(self, train_loader, val_loader):
        """Enhanced ensemble training with better strategies"""
        logger.info("=" * 60)
        logger.info("ENHANCED ENSEMBLE EĞİTİMİ BAŞLIYOR")
        logger.info("=" * 60)

        if len(self.models) == 0:
            logger.error("❌ Hiçbir model oluşturulmadı!")
            return

        # Train models in order of complexity (lighter models first)
        training_order = ['enhanced_cnn', 'efficientnet_b0', 'densenet121', 'resnet50_enhanced']

        for name in training_order:
            if name in self.models:
                logger.info(f"🔥 {name.upper()} ENHANCED EĞİTİMİ")
                logger.info("=" * 40)

                try:
                    history = self.enhanced_train_model(
                        self.models[name], train_loader, val_loader,
                        epochs=CONFIG['EPOCHS'], model_name=name
                    )
                    self.histories[name] = history

                    # Aggressive cleanup after each model
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(2)  # Allow memory to be freed

                except Exception as e:
                    logger.error(f"❌ {name} eğitim hatası: {e}")
                    if name in self.models:
                        del self.models[name]
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue

        logger.info("🎉 Enhanced ensemble eğitimi tamamlandı!")

    def save_enhanced_models(self, save_dir='enhanced_pytorch_models'):
        """Enhanced model saving with metadata"""
        try:
            os.makedirs(save_dir, exist_ok=True)

            # Save ensemble metadata
            ensemble_info = {
                'config': CONFIG,
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'img_size': self.img_size,
                'models': list(self.models.keys()),
                'histories': self.histories
            }

            torch.save(ensemble_info, os.path.join(save_dir, 'ensemble_info.pth'))

            # Save individual models
            for name, model in self.models.items():
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_class': model.__class__.__name__,
                    'num_classes': self.num_classes,
                    'img_size': self.img_size,
                    'config': CONFIG
                }, os.path.join(save_dir, f'{name}_enhanced.pth'))

            logger.info(f"Enhanced modeller {save_dir} dizinine kaydedildi.")

        except Exception as e:
            logger.error(f"Enhanced model kaydetme hatası: {e}")


def enhanced_memory_cleanup():
    """Enhanced memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def main():
    """Enhanced main function with YOLO integration options"""
    logger.info("🧠 ENHANCED Brain Tumor Detection System v3.0 + YOLO")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("ENHANCED BRAIN TUMOR DETECTION SYSTEM v3.0")
    print("=" * 60)
    print("1. Modelleri eğit (Training)")
    print("2. Tek görüntü analizi (Single Image)")
    print("3. Canlı webcam analizi (Live Analysis)")
    print("4. Video dosyası analizi (Video File)")
    print("5. Toplu görüntü analizi (Batch Processing)")
    print("6. YOLO + Classifier entegre test")
    print("0. Çıkış (Exit)")
    print("=" * 60)

    try:
        choice = input("Seçiminizi yapın (0-6): ").strip()

        if choice == '1':
            # Model eğitimi
            train_models()
        elif choice == '2':
            # Tek görüntü
            single_image_analysis()
        elif choice == '3':
            # Webcam
            webcam_analysis()
        elif choice == '4':
            # Video dosyası
            video_file_analysis()
        elif choice == '5':
            # Toplu işlem
            batch_processing()
        elif choice == '6':
            # YOLO entegre test
            yolo_integrated_test()
        elif choice == '0':
            logger.info("Sistem kapatılıyor...")
            return
        else:
            print("Geçersiz seçim!")
            main()

    except KeyboardInterrupt:
        logger.info("⚠️ Kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"❌ Kritik hata: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        enhanced_memory_cleanup()
        logger.info("Memory cleanup tamamlandı")


def train_models():
    """Model eğitimi fonksiyonu"""
    logger.info("=" * 60)
    logger.info("MODEL EĞİTİMİ")
    logger.info("=" * 60)

    # Setup multiprocessing for Windows
    if IS_WINDOWS:
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
            logger.info("Windows multiprocessing configured")
        except RuntimeError as e:
            logger.info(f"Multiprocessing already set: {e}")

    # Enhanced device setup
    device, batch_size = setup_device()

    # Create enhanced ensemble
    ensemble = EnhancedBrainTumorEnsemble(
        device=device,
        batch_size=batch_size,
        num_classes=4,
        img_size=CONFIG['IMG_SIZE']
    )

    # Find data directories
    current_dir = Path.cwd()
    logger.info(f"Current directory: {current_dir}")

    possible_paths = [
        (current_dir / "data" / "Training", current_dir / "data" / "Testing"),
        (current_dir / "yüztanıma" / "data" / "Training", current_dir / "yüztanıma" / "data" / "Testing"),
        (current_dir / "Training", current_dir / "Testing"),
    ]

    train_dir, test_dir = None, None
    for train_path, test_path in possible_paths:
        if train_path.exists() and test_path.exists():
            train_dir, test_dir = train_path, test_path
            break

    if not train_dir:
        logger.error("❌ Data directories not found!")
        logger.info("Required directory structure:")
        logger.info("  data/Training/[glioma, meningioma, notumor, pituitary]")
        logger.info("  data/Testing/[glioma, meningioma, notumor, pituitary]")
        return

    logger.info(f"✅ Data directories found:")
    logger.info(f"  Training: {train_dir}")
    logger.info(f"  Testing: {test_dir}")

    # Enhanced data preparation
    enhanced_memory_cleanup()

    logger.info("📊 Enhanced data loading...")
    train_loader, val_loader, test_loader = ensemble.prepare_enhanced_data(train_dir, test_dir)

    # Create enhanced models
    ensemble.create_enhanced_models()

    if len(ensemble.models) == 0:
        logger.error("❌ No models created!")
        return

    # Enhanced training
    start_time = time.time()
    ensemble.train_enhanced_ensemble(train_loader, val_loader)
    training_time = time.time() - start_time

    logger.info(f"⏱️ Total enhanced training time: {training_time / 60:.1f} minutes")

    # Save enhanced models
    ensemble.save_enhanced_models()

    logger.info("🎉 Model eğitimi tamamlandı!")


def single_image_analysis():
    """Tek görüntü analizi"""
    logger.info("=" * 60)
    logger.info("TEK GÖRÜNTÜ ANALİZİ")
    logger.info("=" * 60)

    device, _ = setup_device()

    # Sistem oluştur
    system = EnhancedBrainTumorSystem(
        yolo_model_path=None,  # Genel YOLO kullanılacak
        classifier_model_dir='enhanced_pytorch_models',
        device=device
    )

    image_path = input("Görüntü dosyası yolunu girin: ").strip()

    if not Path(image_path).exists():
        logger.error(f"Dosya bulunamadı: {image_path}")
        return

    logger.info("Görüntü analiz ediliyor...")
    result = system.process_single_image(image_path, save_results=True)

    if result:
        logger.info("📊 ANALİZ SONUÇLARI:")
        logger.info(f"  Tespit sayısı: {result['num_detections']}")

        for i, res in enumerate(result['results']):
            classification = res['classification']
            logger.info(f"\n  Bölge {i + 1}:")
            logger.info(f"    Sınıf: {classification['class']}")
            logger.info(f"    Güven: {classification['confidence']:.3f}")
            logger.info(f"    Kombine güven: {res['combined_confidence']:.3f}")

        # Görüntüyü göster
        image = cv2.imread(image_path)
        result_image = system.draw_results_on_image(image, result)
        cv2.imshow('Analiz Sonucu', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logger.error("Görüntü analiz edilemedi!")


def webcam_analysis():
    """Webcam canlı analiz"""
    logger.info("=" * 60)
    logger.info("WEBCAM CANLI ANALİZ")
    logger.info("=" * 60)

    device, _ = setup_device()

    system = EnhancedBrainTumorSystem(
        yolo_model_path=None,
        classifier_model_dir='enhanced_pytorch_models',
        device=device
    )

    logger.info("Webcam başlatılıyor... ESC ile çıkış")
    system.process_video('webcam', display=True)


def video_file_analysis():
    """Video dosyası analizi"""
    logger.info("=" * 60)
    logger.info("VIDEO DOSYASI ANALİZİ")
    logger.info("=" * 60)

    device, _ = setup_device()

    system = EnhancedBrainTumorSystem(
        yolo_model_path=None,
        classifier_model_dir='enhanced_pytorch_models',
        device=device
    )

    video_path = input("Video dosyası yolunu girin: ").strip()
    output_path = input("Çıktı video yolu (boş bırakabilirsiniz): ").strip()
    output_path = output_path if output_path else None

    if not Path(video_path).exists():
        logger.error(f"Video dosyası bulunamadı: {video_path}")
        return

    logger.info("Video işleniyor... ESC ile durdurabilirsiniz")
    system.process_video(video_path, output_path, display=True)


def batch_processing():
    """Toplu görüntü işleme"""
    logger.info("=" * 60)
    logger.info("TOPLU GÖRÜNTÜ İŞLEME")
    logger.info("=" * 60)

    device, _ = setup_device()

    system = EnhancedBrainTumorSystem(
        yolo_model_path=None,
        classifier_model_dir='enhanced_pytorch_models',
        device=device
    )

    input_dir = input("Görüntü klasörü yolunu girin: ").strip()
    output_dir = input("Sonuç klasörü (varsayılan: batch_results): ").strip()
    output_dir = output_dir if output_dir else 'batch_results'

    if not Path(input_dir).exists():
        logger.error(f"Klasör bulunamadı: {input_dir}")
        return

    logger.info("Toplu işlem başlatılıyor...")
    system.batch_process(input_dir, output_dir)


def yolo_integrated_test():
    """YOLO entegre sistem testi"""
    logger.info("=" * 60)
    logger.info("YOLO ENTEGRASYONu TEST")
    logger.info("=" * 60)

    if not YOLO_AVAILABLE:
        logger.error("❌ YOLO mevcut değil! pip install ultralytics ile kurun.")
        return

    device, _ = setup_device()

    # Test için basit sistem
    detector = YOLOBrainDetector(device=device)

    # DÜZELTME: model_dir parametresini ekleyin
    classifier = BrainTumorClassifier(model_dir='enhanced_pytorch_models', device=device)

    test_image_path = input("Test görüntüsü yolu: ").strip()

    if not Path(test_image_path).exists():
        logger.error(f"Test görüntüsü bulunamadı: {test_image_path}")
        return

    # Görüntüyü yükle
    image = cv2.imread(test_image_path)

    # YOLO tespiti
    logger.info("YOLO ile tespit yapılıyor...")
    detections = detector.detect_brain_regions(image)
    logger.info(f"YOLO {len(detections)} bölge tespit etti")

    # Her tespit için sınıflandırma
    for i, detection in enumerate(detections):
        logger.info(f"\nBölge {i + 1} analiz ediliyor...")
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        if roi.size > 0:
            result = classifier.predict_single_image(roi)
            if result:
                logger.info(f"  Sınıf: {result['class']}")
                logger.info(f"  Güven: {result['confidence']:.3f}")
                logger.info(f"  Probabilities: {result['probabilities']}")

        # ROI'yi ayrı pencerede göster
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            cv2.imshow(f'ROI {i + 1}', roi)

    # Ana görüntüyü bbox'larla göster
    result_image = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f"Conf: {detection['confidence']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLO Detections', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    if IS_WINDOWS:
        try:
            import multiprocessing

            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    main()
