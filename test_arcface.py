from arcface.nets.arcface import Arcface
from arcface.utils.utils import get_num_classes
import torch

arcface_model_path = "models/arcface.pth"
annotation_path = "arcface/r553.txt"
num_classes = get_num_classes(annotation_path=annotation_path)
backbone = "resnet18"
pretrained = False
arcface_model = Arcface(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
ckpt=torch.load(arcface_model_path, map_location='cpu')
# 
arcface_model.load_state_dict(ckpt)

arcface_model = torch.nn.DataParallel(arcface_model)
print(arcface_model)