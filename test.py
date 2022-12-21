import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(img_size=224, num_classes=15,
                        pretrained_model="pretrained_weight/TimeSformer_divST_8_224_SSv2.pyth")
x = torch.randn(2, 3, 8, 224, 2224)
y = model(x)
print()