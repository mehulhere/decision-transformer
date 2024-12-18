import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        # Ensure returns_to_go has correct shape (batch, seq, 1)
        returns_to_go = rtg[:, :-1].unsqueeze(-1)

        # Forward pass with correct arguments
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask
        )

        action_target = torch.clone(actions)
        act_dim = action_preds.shape[2]

        # Mask and reshape outputs and targets
        valid_indices = attention_mask.reshape(-1) > 0
        action_preds = action_preds.reshape(-1, act_dim)[valid_indices]
        action_target = action_target.reshape(-1, act_dim)[valid_indices]

        # Compute loss
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds - action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
