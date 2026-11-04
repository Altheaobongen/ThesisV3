import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model_architecture import MultiInputBPCNN

LABEL_MAPPING = {
    "Hypotension": 0,
    "Normal": 1,
    "Elevated": 2,
    "Stage 1 Hypertension": 3,
    "Stage 2 Hypertension": 4
}

class BPDataset(Dataset):
    def __init__(self, hemo_data, demo_data, labels, stiffness):
        self.hemo = torch.tensor(hemo_data, dtype=torch.float32).unsqueeze(1)
        self.demo = torch.tensor(demo_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.stiffness = torch.tensor(stiffness, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.hemo[idx], self.demo[idx], self.labels[idx], self.stiffness[idx]

def train():
    df = pd.read_csv("acc_aha_bp_dataset.csv")
    
    # Feature engineering
    map_val = df["dbp"] + (df["sbp"] - df["dbp"]) / 3.0
    df["map"] = map_val
    
    hemo_cols = ["sbp", "dbp", "pulse_pressure", "map"]
    demo_cols = ["age", "sex", "bmi"]
    
    X_hemo = df[hemo_cols].values
    X_demo = df[demo_cols].values
    y_class = df["target_class"].map(LABEL_MAPPING).values
    y_stiff = df[["pwv", "aix"]].values
    
    # Standardization parameters
    hemo_mean, hemo_std = X_hemo.mean(axis=0), X_hemo.std(axis=0) + 1e-7
    demo_mean, demo_std = X_demo.mean(axis=0), X_demo.std(axis=0) + 1e-7
    
    np.savez("scaler_params.npz", 
             hemo_mean=hemo_mean, hemo_std=hemo_std,
             demo_mean=demo_mean, demo_std=demo_std)
    
    X_hemo_norm = (X_hemo - hemo_mean) / hemo_std
    X_demo_norm = (X_demo - demo_mean) / demo_std
    
    X_h_train, X_h_test, X_d_train, X_d_test, y_c_train, y_c_test, y_s_train, y_s_test = train_test_split(
        X_hemo_norm, X_demo_norm, y_class, y_stiff, test_size=0.2, random_state=42, stratify=y_class
    )
    
    train_ds = BPDataset(X_h_train, X_d_train, y_c_train, y_s_train)
    test_ds = BPDataset(X_h_test, X_d_test, y_c_test, y_s_test)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = MultiInputBPCNN(num_classes=5)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    print("Beginning Training...")
    best_acc = 0.0
    for epoch in range(1, 41):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for hemo, demo, labels, stiff in train_loader:
            optimizer.zero_grad()
            logits, stiff_pred = model(hemo, demo)
            
            loss_c = criterion_cls(logits, labels)
            loss_s = criterion_reg(stiff_pred, stiff) * 0.05
            loss = loss_c + loss_s
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
        train_acc = (correct / total) * 100
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for hemo, demo, labels, _ in test_loader:
                logits, _ = model(hemo, demo)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                
        val_acc = (val_correct / val_total) * 100
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "bp_cnn_model.pth")
            
    print(f"Training Complete. Best Validation Accuracy: {best_acc:.2f}%")
    print("Model saved as 'bp_cnn_model.pth'.")

if __name__ == "__main__":
    train()