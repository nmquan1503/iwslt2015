import torch
from tqdm import tqdm

import config

class Trainer:
    def __init__(self, model, train_loader, dev_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.best_dev_loss = float("inf")
        self.train_losses = []
        self.dev_losses = []
        self.start_epoch = 1

        if config.RESUME_TRAINING:
            checkpoint = torch.load(config.LAST_CHECKPOINT_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.train_losses = checkpoint["train_losses"]
            self.dev_losses = checkpoint["dev_losses"]
            self.best_dev_loss = min(self.dev_losses)
            self.start_epoch = len(self.train_losses) + 1
    
    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Train"):
            self.optimizer.zero_grad()

            input_ids = batch["fused_input_ids"].to(self.device)
            target_ids = batch["fused_target_ids"].to(self.device)

            logits = self.model(input_ids, attn_gate_threshold=config.ATTN_GATE_THRESHOLD)

            loss = self.criterion(logits.view(-1, config.VOCAB_SIZE), target_ids.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm(self.dev_loader, desc="Eval"):
            input_ids = batch["fused_input_ids"].to(self.device)
            target_ids = batch["fused_target_ids"].to(self.device)

            logits = self.model(input_ids, attn_gate_threshold=config.ATTN_GATE_THRESHOLD)
            
            loss = self.criterion(logits.view(-1, config.VOCAB_SIZE), target_ids.view(-1))

            total_loss += loss.item()
        
        return total_loss / len(self.dev_loader)

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + config.NUM_EPOCHS):
            print("=" * 10 + f" Epoch {epoch} " + "=" * 10)
            
            train_loss = self._train_one_epoch()
            dev_loss = self._eval()
            self.train_losses.append(train_loss)
            self.dev_losses.append(dev_loss)
            print(f"Train loss: {train_loss:.4f}")
            print(f"Dev loss: {dev_loss:.4f}")

            if dev_loss < self.best_dev_loss:
                self.best_dev_loss = dev_loss
                torch.save(self.model.state_dict(), config.BEST_MODEL_PATH)
                print(">>> Save best model")
            
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "dev_losses": self.dev_losses
            }, config.LAST_CHECKPOINT_PATH)
