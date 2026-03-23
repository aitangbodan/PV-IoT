import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os


class DetectorTrainer:
    def __init__(self, model, device, lr=0.01, epochs=50, save_dir='./checkpoints', log_dir='./logs_detector'):
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir)
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    def train(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            start_time = time.time()
            for i, (images, targets) in enumerate(train_loader):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                train_loss += losses.item()
                if i % 50 == 0:
                    print(f"Epoch {epoch + 1}, Iter {i}, Loss: {losses.item():.4f}")

            avg_loss = train_loss / len(train_loader)
            self.writer.add_scalar('Train/Loss', avg_loss, epoch)

            # 验证
            val_loss = self.validate(val_loader)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)

            # 保存模型
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'detector_epoch{epoch + 1}.pth'))

            print(
                f"Epoch {epoch + 1} done. Time: {time.time() - start_time:.2f}s. Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        return total_loss / len(val_loader)