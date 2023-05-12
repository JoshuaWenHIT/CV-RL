#
#  Created by Joshua Wen on 2023/03/20.
#  Copyright © 2023 Joshua Wen. All rights reserved.
#
import random
import time
import shutil
from loguru import logger

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from dqn_core import make_env, QNetwork, linear_schedule


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save checkpoint model to disk

        state -- checkpoint state: model weight and other info
                 binding by user
        is_best -- if the checkpoint is the best. If it is, then
                   save as a best model
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def load_checkpoint(filename, model):
    """Load previous checkpoint model

       filename -- model file name
       model -- DQN model
    """
    try:
        checkpoint = torch.load(filename)
    except:
        # load weight saved on gpy device to cpu device
        # see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    episode = checkpoint['episode']
    epsilon = checkpoint['epsilon']
    print('pretrained episode = {}'.format(episode))
    print('pretrained epsilon = {}'.format(epsilon))
    model.load_state_dict(checkpoint['state_dict'])
    # time_step = checkpoint.get('best_time_step', None)
    # if time_step is not None:
    #     time_step = checkpoint('time_step')
    # print('pretrained time step = {}'.format(time_step))
    # return episode, epsilon, time_step
    return episode, epsilon


def train_dqn(args):
    run_name = f"{args.env_id}__{args.exp_name}__train__{args.seed}__{int(time.time())}"
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(f"runs/{run_name}/log.txt", format=loguru_format, level="INFO", enqueue=True)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, args.num_layers).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, args.num_layers).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                # print(f"global_step={global_step}, episodic_action={info['episode']['a']}, episodic_return={info['episode']['r']}")
                logger.info(f"global_step={global_step}, episodic_state={info['episode']['n']}, episodic_action={info['episode']['a']}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_action", info["episode"]["a"], global_step)
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                logger.info(f"SPS:{int(global_step / (time.time() - start_time))}")
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            if global_step % args.save_checkpoints_freq == 0:
                save_checkpoint({
                    "episode": global_step,
                    "epsilon": epsilon,
                    "state_dict": q_network.state_dict(),
                }, is_best=False, filename=f"runs/{run_name}/checkpoint_episode_%d.pth" % global_step)
                logger.info("checkpoint saved, episode={}".format(global_step))

    envs.close()
    writer.close()


def test_dqn(model_filename, args):
    run_name = f"{args.env_id}__{args.exp_name}__test__{args.seed}__{int(time.time())}"
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(f"runs/{run_name}/log.txt", format=loguru_format, level="INFO", enqueue=True)
    logger.info("loading checkpoints: {}".format(model_filename))

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model = QNetwork(envs, args.num_layers).to(device)
    load_checkpoint(model_filename, model)

    obs = envs.reset()

    while True:
        q_values = model(torch.Tensor(obs).to(device))
        action = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, rewards, dones, infos = envs.step(action)
        if dones:
            break
        for info in infos:
            if "episode" in info.keys():
                logger.info(f"episodic_state={info['episode']['n']}, episodic_action={info['episode']['a']}, "
                            f"episodic_return={info['episode']['r']}")
                break
