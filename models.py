import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import timm

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("WARNING: segmentation_models_pytorch not installed.")

def build_tumornet_2d(pretrained: bool=True) -> nn.Module:
    if SMP_AVAILABLE:
        return smp.FPN(
            encoder_name="resnet152",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,
            classes=1,
            activation=None,
            decoder_dropout=0.2,
        )
    return _FallbackUNet(1, 1)

def build_tumornet_2d_unetpp(pretrained: bool=True) -> nn.Module:
    if SMP_AVAILABLE:
        return smp.UnetPlusPlus(
            encoder_name="resnet152",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,
            classes=1,
            activation=None,
            decoder_attention_type="scse",
        )
    return _FallbackUNet(1, 1)

class _DoubleConv(nn.Module):
    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _FallbackUNet(nn.Module):
    def __init__(self, ic: int=1, nc: int=1):
        super().__init__()
        f = [64, 128, 256, 512]
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = ic
        for o in f:
            self.enc.append(_DoubleConv(prev, o))
            self.pool.append(nn.MaxPool2d(2))
            prev = o
        self.bot = _DoubleConv(f[-1], f[-1]*2)
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for o in reversed(f):
            self.ups.append(nn.ConvTranspose2d(o*2, o, 2, stride=2))
            self.dec.append(_DoubleConv(o*2, o))
        self.head = nn.Conv2d(f[0], nc, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips, feat = [], x
        for e, p in zip(self.enc, self.pool):
            feat = e(feat)
            skips.append(feat)
            feat = p(feat)
        feat = self.bot(feat)
        for u, d, s in zip(self.ups, self.dec, reversed(skips)):
            feat = u(feat)
            if feat.shape!=s.shape:
                feat = F.interpolate(feat, size=s.shape[2:])
            feat = d(torch.cat([s, feat], 1))
        return self.head(feat)

class DoubleConv3D(nn.Module):
    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm3d(oc), nn.ReLU(True),
            nn.Conv3d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm3d(oc), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResBlock3D(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1, bias=False), nn.BatchNorm3d(c), nn.ReLU(True),
            nn.Conv3d(c, c, 3, padding=1, bias=False), nn.BatchNorm3d(c),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x+self.net(x))

class AttentionGate3D(nn.Module):
    def __init__(self, Fg: int, Fl: int, Fi: int):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv3d(Fg, Fi, 1), nn.BatchNorm3d(Fi))
        self.Wx = nn.Sequential(nn.Conv3d(Fl, Fi, 1), nn.BatchNorm3d(Fi))
        self.psi = nn.Sequential(nn.Conv3d(Fi, 1, 1), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = nn.ReLU(True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x*self.psi(self.relu(self.Wg(g)+self.Wx(x)))

class AttentionUNet3D(nn.Module):
    def __init__(self, ic: int=1, nc: int=1, features: Optional[List[int]]=None):
        super().__init__()
        features = features or [32, 64, 128, 256]
        self.enc = nn.ModuleList()
        self.res = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = ic
        for f in features:
            self.enc.append(DoubleConv3D(prev, f))
            self.res.append(ResBlock3D(f))
            self.pool.append(nn.MaxPool3d((1, 2, 2)))
            prev = f
        self.bot = nn.Sequential(DoubleConv3D(features[-1], features[-1]*2), ResBlock3D(features[-1]*2), nn.Dropout3d(0.1))
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.attn = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose3d(f*2, f, (1, 2, 2), stride=(1, 2, 2)))
            self.dec.append(DoubleConv3D(f*2, f))
            self.attn.append(AttentionGate3D(f*2, f, f//2))
        self.head = nn.Conv3d(features[0], nc, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips, feat = [], x
        for e, r, p in zip(self.enc, self.res, self.pool):
            feat = r(e(feat))
            skips.append(feat)
            feat = p(feat)
        feat = self.bot(feat)
        for up, dec, attn, skip in zip(self.ups, self.dec, self.attn, reversed(skips)):
            feat = up(feat)
            skip = attn(g=feat, x=skip)
            if feat.shape!=skip.shape:
                feat = F.interpolate(feat, size=skip.shape[2:])
            feat = dec(torch.cat([skip, feat], 1))
        return self.head(feat)

class MCDropoutWrapper(nn.Module):
    def __init__(self, model: nn.Module, dropout_p: float=0.1, n_samples: int=10):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self._inject_dropout(self.model, dropout_p)

    def _inject_dropout(self, module: nn.Module, p: float):
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.GELU)):
                setattr(module, name, nn.Sequential(child, nn.Dropout2d(p)))
            else:
                self._inject_dropout(child, p)

    def forward(self, x: torch.Tensor):
        if self.training:
            return self.model(x)
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                preds.append(torch.sigmoid(self.model(x)))
        self.eval()
        preds = torch.stack(preds)
        return preds.mean(0), preds.var(0)

    def predict_with_uncertainty(self, x: torch.Tensor):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                preds.append(torch.sigmoid(self.model(x)))
        self.eval()
        preds = torch.stack(preds)
        return preds.mean(0), preds.var(0)

class BaseStageClassifier(nn.Module):
    def __init__(self, n_classes: int=4, dropout: float=0.3):
        super().__init__()
        self.n_classes = n_classes
        self.dropout = dropout

    def _make_head(self, in_features: int, n_classes: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(512, n_classes),
        )

class ResNet50Classifier(BaseStageClassifier):
    def __init__(self, n_classes: int=4, pretrained: bool=True, dropout: float=0.3):
        super().__init__(n_classes, dropout)
        self.backbone = timm.create_model("resnet50", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = self._make_head(self.backbone.num_features, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

class ResNet152Classifier(BaseStageClassifier):
    def __init__(self, n_classes: int=4, pretrained: bool=True, dropout: float=0.3):
        super().__init__(n_classes, dropout)
        self.backbone = timm.create_model("resnet152", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = self._make_head(self.backbone.num_features, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

class DenseNet121Classifier(BaseStageClassifier):
    def __init__(self, n_classes: int=4, pretrained: bool=True, dropout: float=0.3):
        super().__init__(n_classes, dropout)
        self.backbone = timm.create_model("densenet121", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = self._make_head(self.backbone.num_features, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

class SwinTransformerClassifier(BaseStageClassifier):
    def __init__(self, n_classes: int=4, pretrained: bool=True, dropout: float=0.3):
        super().__init__(n_classes, dropout)
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0)
        self.head = self._make_head(self.backbone.num_features, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

class EfficientNetV2Classifier(BaseStageClassifier):
    def __init__(self, n_classes: int=4, pretrained: bool=True, dropout: float=0.3):
        super().__init__(n_classes, dropout)
        self.backbone = timm.create_model("efficientnetv2_s", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = self._make_head(self.backbone.num_features, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

class ConvNeXtClassifier(BaseStageClassifier):
    def __init__(self, n_classes: int=4, pretrained: bool=True, dropout: float=0.3):
        super().__init__(n_classes, dropout)
        self.backbone = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = self._make_head(self.backbone.num_features, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

class ViTClassifier(BaseStageClassifier):
    def __init__(self, n_classes: int=4, pretrained: bool=True, dropout: float=0.3):
        super().__init__(n_classes, dropout)
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
        self.head = self._make_head(self.backbone.num_features, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

class MultiScaleCTClassifier(BaseStageClassifier):
    def __init__(self, n_classes: int=4, pretrained: bool=True, dropout: float=0.3):
        super().__init__(n_classes, dropout)
        self.enc_small = timm.create_model("efficientnetv2_s", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.enc_medium = timm.create_model("efficientnetv2_s", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.enc_large = timm.create_model("efficientnetv2_s", pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.enc_small.num_features
        self.fusion = nn.Sequential(nn.Linear(feat_dim*3, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, n_classes))
        self.scales = [112, 224, 320]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for enc, size in zip([self.enc_small, self.enc_medium, self.enc_large], self.scales):
            feats.append(enc(F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)))
        return self.fusion(torch.cat(feats, dim=1))

CLASSIFIER_REGISTRY = {
    "resnet50":        ResNet50Classifier,
    "resnet152":       ResNet152Classifier,
    "densenet121":     DenseNet121Classifier,
    "swin_transformer":SwinTransformerClassifier,
    "efficientnetv2":  EfficientNetV2Classifier,
    "convnext":        ConvNeXtClassifier,
    "vit":             ViTClassifier,
    "multiscale_ct":   MultiScaleCTClassifier,
}

SEGMENTATION_REGISTRY = {
    "tumornet_2d": build_tumornet_2d,
    "tumornet_2d_pp": build_tumornet_2d_unetpp,
    "tumornet_3d": lambda p: AttentionUNet3D(1, 1),
}

def get_classifier(name: str, n_classes: int=4, pretrained: bool=True, dropout: float=0.3, **kwargs) -> nn.Module:
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier: {name}. Choose from {list(CLASSIFIER_REGISTRY.keys())}")
    return CLASSIFIER_REGISTRY[name](n_classes=n_classes, pretrained=pretrained, dropout=dropout, **kwargs)

def get_segmentation_model(name: str, pretrained: bool=True, mc_dropout: bool=False, n_samples: int=10) -> nn.Module:
    if name not in SEGMENTATION_REGISTRY:
        raise ValueError(f"Unknown seg model: {name}")
    model = SEGMENTATION_REGISTRY[name](pretrained)
    if mc_dropout:
        model = MCDropoutWrapper(model, n_samples=n_samples)
    return model

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
