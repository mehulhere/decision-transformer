import gym
import numpy as np
import torch
import wandb

import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    # Initialize the environment based on env_name
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    # elif env_name == 'reacher2d':
    #     from decision_transformer.envs.reacher_2d import Reacher2dEnv
    #     env = Reacher2dEnv()
    #     max_ep_len = 100
    #     env_targets = [76, 40]
    #     scale = 10.
    else:
        raise NotImplementedError(f"Environment '{env_name}' is not implemented.")

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # Save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # Used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # Only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # Used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps_list, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # Get sequences from dataset
            s_seq = traj['observations'][si:si + max_len].reshape(1, -1, state_dim)
            a_seq = traj['actions'][si:si + max_len].reshape(1, -1, act_dim)
            r_seq = traj['rewards'][si:si + max_len].reshape(1, -1, 1)
            if 'terminals' in traj:
                d_seq = traj['terminals'][si:si + max_len].reshape(1, -1)
            else:
                d_seq = traj['dones'][si:si + max_len].reshape(1, -1)
            timesteps_seq = np.arange(si, si + s_seq.shape[1]).reshape(1, -1)
            timesteps_seq[timesteps_seq >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg_seq = discount_cumsum(traj['rewards'][si:], gamma=1.)[:s_seq.shape[1] + 1].reshape(1, -1, 1)
            if rtg_seq.shape[1] <= s_seq.shape[1]:
                rtg_seq = np.concatenate([rtg_seq, np.zeros((1, 1, 1))], axis=1)

            # Padding and state + reward normalization
            tlen = s_seq.shape[1]
            s_padded = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s_seq], axis=1)
            s_padded = (s_padded - state_mean) / state_std
            a_padded = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a_seq], axis=1)
            r_padded = np.concatenate([np.zeros((1, max_len - tlen, 1)), r_seq], axis=1)
            d_padded = np.concatenate([np.ones((1, max_len - tlen)) * 2, d_seq], axis=1)
            rtg_padded = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg_seq], axis=1) / scale
            timesteps_padded = np.concatenate([np.zeros((1, max_len - tlen)), timesteps_seq], axis=1)
            mask_seq = np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1)

            s.append(s_padded)
            a.append(a_padded)
            r.append(r_padded)
            d.append(d_padded)
            rtg.append(rtg_padded)
            timesteps_list.append(timesteps_padded)
            mask.append(mask_seq)

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps_list, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    # Initialize the model based on model_type
    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not implemented.")

    model = model.to(device=device)

    # Initialize optimizer and scheduler
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    # Initialize the trainer based on model_type
    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    # Initialize Weights & Biases for logging
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # Uncomment if needed and fixed

    # Training loop
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

    # Close the environment after training
    env.close()


if __name__ == '__main__':
    # Define the list of environments to test
    TEST_ENVIRONMENTS = ['hopper', 'halfcheetah', 'walker2d']

    # Define test parameters
    TEST_VARIANT = {
        'env': '',  # To be set per environment
        'dataset': 'medium',  # medium, medium-replay, medium-expert, expert
        'mode': 'normal',  # normal for standard setting, delayed for sparse
        'K': 20,
        'pct_traj': 0.1,  # Use a subset of trajectories
        'batch_size': 16,  # Smaller batch size for quicker runs
        'model_type': 'dt',  # dt for decision transformer, bc for behavior cloning
        'embed_dim': 128,
        'n_layer': 3,
        'n_head': 1,
        'activation_function': 'relu',
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'warmup_steps': 10000,
        'num_eval_episodes': 10,  # Fewer evaluation episodes
        'max_iters': 2,  # Minimal number of iterations
        'num_steps_per_iter': 1000,  # Minimal steps per iteration
        'device': 'cuda',  # Change to 'cpu' if GPU is not available
        'log_to_wandb': False,  # Set to True if logging is desired
    }

    # Iterate over each environment and run the experiment
    for env in TEST_ENVIRONMENTS:
        print(f"\n{'='*50}\nStarting test for environment: {env}\n{'='*50}")
        # Update the environment in the variant
        TEST_VARIANT['env'] = env

        # Generate a unique experiment prefix for each environment
        experiment_prefix = 'gym-test-experiment'

        # Run the experiment
        experiment(experiment_prefix, variant=TEST_VARIANT)

        print(f"Completed test for environment: {env}\n{'='*50}\n")

    print("All tests completed successfully.")
