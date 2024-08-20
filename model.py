import torchvision.models as models
import torch
import torch.nn as nn

class DSarcNet(nn.Module):
    def __init__(self, num_features, emb_size=768):
        super(DSarcNet, self).__init__()
        
        convnext_tiny1 = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        convnext_block1 = list(convnext_tiny1.children())[0][0:2]
        convnext_block2 = list(convnext_tiny1.children())[0][2:4]
        convnext_block3 = list(convnext_tiny1.children())[0][4:6]
        convnext_block4 = list(convnext_tiny1.children())[0][6:8]
        
        self.convnextblock1 = nn.Sequential(*convnext_block1)
        self.convnextblock2 = nn.Sequential(*convnext_block2)
        self.convnextblock3 = nn.Sequential(*convnext_block3)
        self.convnextblock4 = nn.Sequential(*convnext_block4)
        
        weights_backbone = models.Swin_V2_T_Weights.DEFAULT
        weights_backbone = models.Swin_V2_T_Weights.verify(weights_backbone)
        
        cnn_model2 = models.swin_v2_t(weights=weights_backbone, progress=True)
        cnn_modules2 = list(list(cnn_model2.children())[0])
        self.cnn_modules2 = nn.ModuleList(cnn_modules2[i].to(DEVICE) for i in range(0, len(cnn_modules2)))
        
        self.post_add_conv1 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.Conv2d(96, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.post_add_conv2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.post_add_conv3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.Conv2d(384, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.post_add_conv4 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.Conv2d(768, 768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        
        self.down1 = nn.Sequential(nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1))
        self.down3 = nn.Sequential(nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1))
        
        self.down1f = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.down2f = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fc = self.fcClassifier(2112, 1)

    def forward(self, x, feature, den_matrix):
        feature_matrix = den_matrix[:, :3, :, :]
        den_matrix = den_matrix[:, -1, :, :][:, None, :, :]
        
        x_patch = self.cnn_modules2[0](feature_matrix)
        # x_patch = x_patch * transforms.Resize((56, 56))(den_matrix).permute(0, 2, 3, 1)
        
        x_patch = self.cnn_modules2[1](x_patch)
        x_conv = self.convnextblock1(x)
        x_conv2 = self.convnextblock1(feature_matrix)
        # x_add1 = x_patch.permute(0, 3, 1, 2) + x_conv
        x_add1 = x_conv + x_conv2
        x_add1 = self.post_add_conv1(x_add1)
        
        x_patch = self.cnn_modules2[2](x_patch)
        # x_patch = x_patch * transforms.Resize((28, 28))(den_matrix).permute(0, 2, 3, 1)
        
        x_patch = self.cnn_modules2[3](x_patch)
        x_conv = self.convnextblock2(x_conv)
        x_conv2 = self.convnextblock2(x_conv2)
        # x_add2 = x_patch.permute(0, 3, 1, 2) + x_conv
        x_add2 = x_conv2 + x_conv
        x_add2 = self.post_add_conv2(x_add2)
        
        x_patch = self.cnn_modules2[4](x_patch)
        # x_patch = x_patch * transforms.Resize((14, 14))(den_matrix).permute(0, 2, 3, 1)
        
        x_patch = self.cnn_modules2[5](x_patch)
        x_conv = self.convnextblock3(x_conv)
        x_conv2 = self.convnextblock3(x_conv2)
        # x_add3 = x_patch.permute(0, 3, 1, 2) + x_conv
        x_add3 = x_conv2 + x_conv
        x_add3 = self.post_add_conv3(x_add3)
        # print(x_add3.shape)
        
        x_patch = self.cnn_modules2[6](x_patch)
        #x_patch = x_patch * transforms.Resize((7, 7))(den_matrix).permute(0, 2, 3, 1)
        
        x_patch = self.cnn_modules2[7](x_patch)
        x_conv = self.convnextblock4(x_conv)
        x_conv2 = self.convnextblock4(x_conv2)
        # x_add4 = x_patch.permute(0, 3, 1, 2) + x_conv
        x_add4 = x_conv2 + x_conv
        x_add4 = self.post_add_conv4(x_add4)
        
        x_add1 = self.down1(x_add1)
        x_add2 = x_add2 + x_add1
        print(x_add2.shape)
        x_add2 = self.down2(x_add2)
        x_add3 = x_add3 + x_add2
        print(x_add3.shape)
        x_add3 = self.down3(x_add3)
        x_add4 = x_add4 + x_add3
        
        x_add1 = self.down1f(x_add1)
        x_add2 = self.down2f(x_add2)
        
        x_pooled = torch.cat((x_add1, x_add2, x_add3, x_add4), dim=1)  # torch.Size([40, ...])
        x_pooled = torch.mean(x_pooled, dim=[2, 3])
        x_pooled = x_pooled.view(x_pooled.shape[0], -1)
        x = self.fc(x_pooled)
        
        return x

    def fcClassifier(self, inChannels, numClasses):
        fc_classifier = nn.Sequential(
            nn.Linear(inChannels, 512),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, numClasses)
        )
        return fc_classifier
