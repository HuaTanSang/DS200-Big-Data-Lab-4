import torch
import torch.nn as nn
import torch.optim as optim
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.ml.linalg import VectorUDT
from compute_score import compute_scores

class Trainer:
    def __init__(self, model, sql_context):
        self.model = model
        self.sql = sql_context
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.true_labels = []
        self.pred_labels = []

    def train(self, rdd):
        if rdd.isEmpty():
            return

        # Tạo DataFrame từ RDD
        schema = StructType([
            StructField("features", VectorUDT(), True),
            StructField("label", IntegerType(), True)
        ])
        df = self.sql.createDataFrame(rdd, schema)

        # Collect về driver
        data = df.collect()
        feats = [row['features'] for row in data]
        lbls  = [row['label'] for row in data]

        # Torch tensor
        x = torch.tensor(feats).view(-1, 1, 28, 28).float()
        y = torch.tensor(lbls).long()

        # Forward – backward
        out = self.model(x)
        loss = self.criterion(out, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Dự đoán và ghi lại
        preds = out.argmax(dim=1).tolist()
        self.true_labels.extend(lbls)
        self.pred_labels.extend(preds)

        # Tính score tạm thời
        scores = compute_scores(preds, lbls)
        print(f"[Batch] Loss={loss.item():.4f}  Acc={scores['accuracy']:.4f}  F1={scores['f1']:.4f}")
