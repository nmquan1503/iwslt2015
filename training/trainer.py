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
    
    def _compute_aux_loss(self, hard_gates, attn_weights, valid_masks):
        total_loss = 0.0
        total_count = 0

        for hard_gate, attn_weight, valid_mask in zip(hard_gates, attn_weights, valid_masks):
            gate = hard_gate[:, 0, :, :]                     # (B, L, L)
            max_attn = attn_weight.max(dim=1).values.detach() # (B, L, L)
            mask_all = valid_mask[:, 0, :, :].bool()         # (B, L, L)
            S = mask_all.float().sum(dim=-1, keepdim=True).clamp(min=1)  # (B, L, 1)
            scaled_attn = S * max_attn                       # (B, L, L)
            L = gate.shape[-1]
            diag_mask = torch.eye(L, device=gate.device, dtype=torch.bool).unsqueeze(0)
            mask_no_diag = mask_all & (~diag_mask)
            masked_for_max = scaled_attn.masked_fill(~mask_no_diag, float('-inf'))
            max_per_key = masked_for_max.max(dim=1).values   # (B, L)
            has_query = mask_no_diag.any(dim=1)              # (B, L)
            max_per_key = torch.where(has_query, max_per_key, torch.tensor(1.0, device=gate.device))
            penalty_per_key = torch.relu(1.0 - max_per_key)  # (B, L)
            penalty_matrix = penalty_per_key.unsqueeze(1)    # (B, 1, L)
            loss_element = gate * mask_no_diag.float() * penalty_matrix  # (B, L, L)
            valid_keys = mask_all.any(dim=1).float()         # (B, L)
            per_key_total = loss_element.sum(dim=1)          # (B, L)
            sample_loss = (per_key_total * valid_keys).sum(dim=1)  # (B,) 
            sample_count = valid_keys.sum(dim=1)              # (B,)

            total_loss += sample_loss.sum()
            total_count += sample_count.sum()

        return total_loss / total_count.clamp(min=1)

    def _train_one_epoch(self):
        self.model.train()
        total_target_loss = 0.0
        total_gate_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Train"):
            self.optimizer.zero_grad()

            input_ids = batch["fused_input_ids"].to(self.device)
            target_ids = batch["fused_target_ids"].to(self.device)
            fused_lengths = batch["fused_lengths"].to(self.device)

            logits, hard_gates, attn_weights, valid_masks = self.model(input_ids)

            target_loss = self.criterion(logits.view(-1, config.VOCAB_SIZE), target_ids.view(-1))
            gate_loss = self._compute_aux_loss(hard_gates, attn_weights, valid_masks)

            loss = target_loss + config.LAMBDA * gate_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_target_loss += target_loss.item()
            total_gate_loss += gate_loss

        avg_target_loss = total_target_loss / len(self.train_loader)
        avg_gate_loss = total_gate_loss / len(self.train_loader)

        print(f"Train target loss: {avg_target_loss:.4f} | Gate loss: {avg_gate_loss}")

        return total_target_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        total_target_loss = 0.0
        for batch in tqdm(self.dev_loader, desc="Eval"):
            input_ids = batch["fused_input_ids"].to(self.device)
            target_ids = batch["fused_target_ids"].to(self.device)
            fused_lengths = batch["fused_lengths"].to(self.device)

            logits, _, _, _ = self.model(input_ids)
            
            target_loss = self.criterion(logits.view(-1, config.VOCAB_SIZE), target_ids.view(-1))

            total_target_loss += target_loss.item()
        
        avg_target_loss = total_target_loss / len(self.dev_loader)

        print(f"Dev target loss: {avg_target_loss:.4f}")

        return avg_target_loss

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + config.NUM_EPOCHS):
            print("=" * 10 + f" Epoch {epoch} " + "=" * 10)
            
            train_loss = self._train_one_epoch()
            dev_loss = self._eval()
            self.train_losses.append(train_loss)
            self.dev_losses.append(dev_loss)

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
