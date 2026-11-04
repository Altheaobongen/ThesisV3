import torch
import torch.nn as nn

class MultiInputBPCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiInputBPCNN, self).__init__()
        
        # 1D Convolutional Branch for Hemodynamic Feature Vector (Channel=1, Length=4)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(2),
            nn.Flatten()  # 32 * 2 = 64
        )
        
        # MLP Branch for Demographic Features (Age, Sex, BMI) -> 3 inputs
        self.demo_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Fusion Classifier (64 + 16 = 80 inputs)
        self.fusion = nn.Sequential(
            nn.Linear(80, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Task 1: Hypertension Classification Head
        self.classifier = nn.Linear(32, num_classes)
        
        # Task 2: Arterial Stiffness Regression Head (PWV, AIx)
        self.stiffness_regressor = nn.Linear(32, 2)

    def forward(self, hemodyn, demo):
        # hemodyn shape: (batch_size, 1, 4)
        # demo shape: (batch_size, 3)
        feat_conv = self.conv1d(hemodyn)
        feat_demo = self.demo_mlp(demo)
        
        fused = torch.cat([feat_conv, feat_demo], dim=1)
        fused_rep = self.fusion(fused)
        
        class_logits = self.classifier(fused_rep)
        stiffness_preds = self.stiffness_regressor(fused_rep)
        
        return class_logits, stiffness_preds