import os
import sys
import time
import signal
import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from agent import ActorCritic
from checkpoint_manager import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Official 8-Worker Parallel SOTA PPO Trainer for LunarLanderContinuous-v3")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--num-steps", type=int, default=512, help="Steps per environment per rollout (total = num_envs * num_steps)")
    parser.add_argument("--total-timesteps", type=int, default=600000, help="Total environment steps to train (default: 600,000)")
    parser.add_argument("--lr-start", type=float, default=6e-5, help="Fine-tuning initial learning rate")
    parser.add_argument("--lr-end", type=float, default=1e-5, help="Final annealed learning rate")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.98, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy coefficient for attitude exploration")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO optimization epochs per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size for PPO updates")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/latest_checkpoint_ep4000_177mean_backup.pth",
        help="Checkpoint path to initialize weights from",
    )
    return parser.parse_args()


def make_env(seed):
    def thunk():
        env = gym.make(
            "LunarLander-v3",
            continuous=True,
            enable_wind=True,
            wind_power=20.0,
            turbulence_power=2.0,
        )
        env.reset(seed=seed)
        return env
    return thunk


def main():
    args = parse_args()
    print("=" * 78)
    print("  PARALLEL SOTA PPO TRAINER: 8-WORKER STORM STABILIZER")
    print("=" * 78)
    print(f" [PARALLEL] Running {args.num_envs} Simultaneous Environments across Diverse Storms")
    print(" [ENV SPEC] Continuous=True | Wind=20.0 (Gale) | Turbulence=2.0 (Torque)")
    print(f" [ROLLOUT]  {args.num_envs} envs x {args.num_steps} steps = {args.num_envs * args.num_steps} steps per update")
    print(f" [LR RANGE] {args.lr_start} -> {args.lr_end} (Polishing Mode)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" [COMPUTE]  PyTorch Device: {device}\n")

    # Initialize 8 Parallel Environments
    envs = gym.vector.SyncVectorEnv([make_env(seed=42 + i * 1000) for i in range(args.num_envs)])

    network = ActorCritic(state_dim=8, action_dim=2).to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr_start, eps=1e-5)

    best_score = -9999.0
    episodes_done = 0
    scores_history = deque(maxlen=100)
    if os.path.exists(args.checkpoint):
        print(f" [CHECKPOINT] Initializing from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        network.load_state_dict(ckpt["network_state_dict"])
        best_score = ckpt.get("best_score", 286.0)
        episodes_done = ckpt.get("episode", 4000)
        print(f" [INITIALIZED] Champion Weights Loaded! (All-Time Best: {best_score:.1f})\n")
    else:
        print(f" [WARN] Checkpoint '{args.checkpoint}' not found. Initializing with clean weights.\n")

    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\n [INTERRUPT] Caught exit signal. Saving parallel checkpoints before shutdown...")
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)

    # Rollout Storage Buffers
    obs = np.zeros((args.num_steps, args.num_envs, 8), dtype=np.float32)
    actions = np.zeros((args.num_steps, args.num_envs, 2), dtype=np.float32)
    logprobs = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    rewards = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    dones = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    values = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    env_returns = np.zeros(args.num_envs, dtype=np.float32)
    env_lengths = np.zeros(args.num_envs, dtype=np.int32)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=42)
    next_done = np.zeros(args.num_envs, dtype=np.float32)

    total_iterations = args.total_timesteps // (args.num_envs * args.num_steps)
    print(f"{'Iter':>5} | {'Steps':>8} | {'Ep Total':>8} | {'Last Flight':>12} | {'Mean (100)':>11} | {'All Best':>8} | {'Result':<18}")
    print("-" * 84)

    try:
        for iteration in range(1, total_iterations + 1):
            if interrupted:
                break

            # Anneal Learning Rate
            frac = (iteration - 1) / max(1, total_iterations - 1)
            lr_now = args.lr_start - frac * (args.lr_start - args.lr_end)
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(args.lr_end, lr_now)

            last_flight_score = 0.0
            last_flight_outcome = "IN FLIGHT"

            # Collect Rollouts from all 8 parallel environments
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    action_t, logprob_t, _, value_t = network.get_action_and_value(obs_t)

                actions[step] = action_t.cpu().numpy()
                logprobs[step] = logprob_t.cpu().numpy()
                values[step] = value_t.squeeze(-1).cpu().numpy()

                env_action = np.clip(actions[step], -1.0, 1.0)
                next_obs, reward, terminated, truncated, _ = envs.step(env_action)
                rewards[step] = reward
                next_done = np.logical_or(terminated, truncated).astype(np.float32)

                env_returns += reward
                env_lengths += 1

                for i in range(args.num_envs):
                    if next_done[i]:
                        episodes_done += 1
                        sc = float(env_returns[i])
                        scores_history.append(sc)
                        last_flight_score = sc

                        if sc >= 200.0:
                            last_flight_outcome = "SOLVED TOUCHDOWN!"
                        elif sc > 50.0:
                            last_flight_outcome = "SOFT TOUCHDOWN"
                        else:
                            last_flight_outcome = "CRASH / TILTED"

                        if sc > best_score:
                            best_score = sc
                            # Save new record
                            torch.save(
                                {
                                    "episode": episodes_done,
                                    "total_steps": global_step,
                                    "score": sc,
                                    "avg_score": float(np.mean(scores_history)),
                                    "best_score": best_score,
                                    "network_state_dict": network.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                },
                                "checkpoints/parallel_best_model.pth",
                            )

                        env_returns[i] = 0.0
                        env_lengths[i] = 0

            # Compute Generalized Advantage Estimation (GAE) across all 8 parallel streams
            with torch.no_grad():
                next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                next_val = network.get_value(next_obs_t).squeeze(-1).cpu().numpy()
                advantages = np.zeros_like(rewards)
                lastgaelam = np.zeros(args.num_envs, dtype=np.float32)

                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_val
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]

                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

                returns = advantages + values

            b_obs = torch.as_tensor(obs.reshape((-1, 8)), dtype=torch.float32, device=device)
            b_logprobs = torch.as_tensor(logprobs.reshape(-1), dtype=torch.float32, device=device)
            b_actions = torch.as_tensor(actions.reshape((-1, 2)), dtype=torch.float32, device=device)
            b_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
            b_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
            b_values = torch.as_tensor(values.reshape(-1), dtype=torch.float32, device=device)

            # Advantage Normalization
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # PPO Minibatch Updates
            batch_indices = np.arange(b_obs.shape[0])
            for epoch in range(args.update_epochs):
                np.random.shuffle(batch_indices)
                for start in range(0, b_obs.shape[0], args.batch_size):
                    end = start + args.batch_size
                    mb_idx = batch_indices[start:end]

                    _, newlogprob, entropy, newvalue = network.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])
                    logratio = newlogprob - b_logprobs[mb_idx]
                    ratio = torch.exp(logratio)

                    # Clipped Surrogate Objective
                    mb_adv = b_advantages[mb_idx]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    v_loss_unclipped = (newvalue.squeeze(-1) - b_returns[mb_idx]) ** 2
                    v_clipped = b_values[mb_idx] + torch.clamp(newvalue.squeeze(-1) - b_values[mb_idx], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                    optimizer.step()

            avg_100 = float(np.mean(scores_history)) if scores_history else 0.0
            print(
                f"{iteration:>5d} | {global_step:>8d} | {episodes_done:>8d} | {last_flight_score:>12.1f} | "
                f"{avg_100:>11.1f} | {best_score:>8.1f} | {last_flight_outcome:<18}",
                flush=True,
            )

            # Save latest checkpoint periodically
            if iteration % 5 == 0:
                torch.save(
                    {
                        "episode": episodes_done,
                        "total_steps": global_step,
                        "score": last_flight_score,
                        "avg_score": avg_100,
                        "best_score": best_score,
                        "network_state_dict": network.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    "checkpoints/parallel_latest.pth",
                )

    finally:
        print("\n [SAVING] Writing final parallel checkpoints to disk...", flush=True)
        torch.save(
            {
                "episode": episodes_done,
                "total_steps": global_step,
                "score": float(np.mean(scores_history)) if scores_history else 0.0,
                "avg_score": float(np.mean(scores_history)) if scores_history else 0.0,
                "best_score": best_score,
                "network_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "checkpoints/parallel_latest.pth",
        )
        envs.close()
        elapsed = time.time() - start_time
        fps = int(global_step / max(1, elapsed))
        print(f" [FINISHED] Trained {global_step} steps across {episodes_done} flights in {elapsed:.1f}s ({fps} FPS).")
        print(" [SAVED] Checkpoints available at: checkpoints/parallel_best_model.pth & checkpoints/parallel_latest.pth\n")


if __name__ == "__main__":
    main()

