from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import v2
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

data_dir = Path("dataset")  # train/, valid/, test/

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
valid_dataset = datasets.ImageFolder(data_dir / "valid", transform=eval_transform)
test_dataset  = datasets.ImageFolder(data_dir / "test",  transform=eval_transform)

n_classes = len(train_dataset.classes)

batch_size = 32
num_workers = 2


print("CLasses:", train_dataset.classes)
print("Train:", len(train_dataset), "Valid:", len(valid_dataset), "Test:", len(test_dataset))



cutmix = v2.CutMix(num_classes=n_classes, alpha=1.0)
mixup = v2.Mixup(num_classes=n_classes, alpha=0.2)

cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):

    images, labels = torch.utils.data.default_collate(batch)
    
    return cutmix_or_mixup(images, labels)


BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn
)

valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)





def build_cnn_model(model_name, strategy="feature_extractor", num_classes=2):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        num_ftrs = model.classifier[2].in_features

    if strategy == "feature_extractor" or strategy == "combined":
        for param in model.parameters():
            param.requires_grad = False
    head = nn.Linear(num_ftrs, num_classes)
    
    if model_name == "resnet50":
        model.fc = head
    elif model_name == "efficientnet_b0":
        model.classifier[1] = head
    elif model_name == "convnext_tiny":
        model.classifier[2] = head

    return model


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        features = self.backbone(x) 
        return self.head(features)

dinov2_model = DINOv2Classifier()

writer = SummaryWriter(log_dir="tboard_logs/ResNet50_FineTuned")