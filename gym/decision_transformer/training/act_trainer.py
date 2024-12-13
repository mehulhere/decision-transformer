import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class ActTrainer(Trainer):

    def train_step(self):
        # Retrieve a batch of data
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_batch(self.batch_size)

        # Prepare targets
        state_target = torch.clone(states)
        action_target = torch.clone(actions)
        reward_target = torch.clone(rewards)

        # Generate timesteps for the current batch
        # timesteps is a sequence of indices [0, 1, 2, ..., seq_len-1] for each sample in the batch
        batch_size, seq_len = states.shape[0], states.shape[1]
        timesteps = torch.arange(seq_len, dtype=torch.long, device=states.device).unsqueeze(0).repeat(batch_size, 1)

        # Forward pass through the model
        # Note: The model expects returns_to_go as `rtg`, not `target_return`.
        # Ensure rtg is shaped correctly as (batch, seq, 1) if needed by the model.
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg, timesteps, attention_mask=attention_mask
        )

        # Reshape action predictions and targets for the loss function
        # action_preds: (batch, seq, act_dim)
        act_dim = action_preds.shape[2]
        # Flatten action_preds and select last timestep of action_target
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:, -1].reshape(-1, act_dim)

        # Compute the loss
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
