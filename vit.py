import timm

class visionTransformer(timm.models.vision_transformer):



    def forward(self, x):
        x =