"""QUANTIZED MODEL BELOW"""

import numpy as np
import torch
import torch.nn as nn

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel

class DecisionTransformer(nn.Module):
    """
    Decision Transformer using Quantized Llama-3 as the base model
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size=4096,  # Updated to match Llama-3 default = 4096
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            quantization_config=None,
            num_hidden_layers=3,  # Set to 12 layers
            **kwargs
    ):
        super().__init__()

        # Add max_length as an attribute
        self.max_length = max_length

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        # Load  Llama-3 model with custom number of layers
        self.transformer = AutoModel.from_pretrained(
            "unsloth/Llama-3.2-1B",
            device_map="auto"
        )

        # Update the model config to use 12 layers
        self.transformer.config.num_hidden_layers = num_hidden_layers

        # Freeze transformer parameters
        # for param in self.transformer.parameters():
        #     param.requires_grad = False

        # Embedding layers 
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # Prediction heads
        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # Embed each modality 
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Add time embeddings 
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack inputs 
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Prepare stacked attention mask
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # Llama transformer processing
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask.bool(),
        )
        x = transformer_outputs.last_hidden_state

        # Reshape x 
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # Reshape inputs
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # Handle max length if specified
        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # Pad tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            
            states = torch.cat([
                torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), 
                states
            ], dim=1).to(dtype=torch.float32)
            
            actions = torch.cat([
                torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), 
                actions
            ], dim=1).to(dtype=torch.float32)
            
            returns_to_go = torch.cat([
                torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), 
                returns_to_go
            ], dim=1).to(dtype=torch.float32)
            
            timesteps = torch.cat([
                torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), 
                timesteps
            ], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        # Get action prediction
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
