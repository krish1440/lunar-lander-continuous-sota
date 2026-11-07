import os
import sys
import signal
import time
import argparse
from collections import deque
import numpy as np
import pygame
import gymnasium as gym

from agent import PPOAgent
from checkpoint_manager import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Official Gymnasium LunarLanderContinuous-v3 SOTA Training")
    parser.add_argument("--episodes", type=int, default=1000, help="Total episodes to train")
    parser.add_argument("--render", action="store_true", default=True, help="Display live game window")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Run in fast headless mode")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last checkpoint")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh from Episode 1")
    parser.add_argument("--save-freq", type=int, default=20, help="Save checkpoint every N episodes")
    parser.add_argument("--update-steps", type=int, default=1024, help="Rollout steps per PPO update (RL-Zoo standard: 1024)")
    parser.add_argument("--lr-start", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--lr-end", type=float, default=5e-5, help="Final annealed learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("  OFFICIAL GYMNASIUM LUNARLANDER-V3 (RL-ZOO GOLD STANDARD CONTINUOUS PPO)")
    print("=" * 75)
    print(" [ENV] Official Gymnasium 'LunarLander-v3' (Uncheatable Benchmark Rules)")
    print(" [SPEC] Continuous=True | Wind=True (20.0 Gale Force) | Turbulence=2.0 (Torque)")
    print(" [ALGO] 64-Neuron Fast-Learning Architecture | Gamma=0.999 | Lambda=0.98 | Epochs=4")
    print(f" [MODE] {'Live Visual Window with Hotkeys' if args.render else 'Fast Headless Mode'}")
    if args.render:
        print(" [KEYS] [Space]=Pause/Resume  [V]=Toggle Fast/Watch  [S]=Save  [Q]=Quit\n")

    render_mode = "human" if args.render else None
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        enable_wind=True,
        wind_power=20.0,
        turbulence_power=2.0,
        render_mode=render_mode,
    )

    if args.render:
        pygame.font.init()
        hud_font = pygame.font.SysFont("Consolas", 13, bold=True)
        key_font = pygame.font.SysFont("Consolas", 12)

    agent = PPOAgent(
        state_dim=8,
        action_dim=2,
        lr=args.lr_start,
        gamma=0.999,
        gae_lambda=0.98,
        update_epochs=4,
        batch_size=64,
    )
    checkpoint_mgr = CheckpointManager(checkpoint_dir="checkpoints")

    start_episode = 1
    total_env_steps = 0
    best_score = -9999.0
    reward_history = []
    recent_scores = deque(maxlen=50)

    # Auto-resume 
    if args.resume and checkpoint_mgr.has_checkpoint():
        ckpt = checkpoint_mgr.load_checkpoint(agent)
        if ckpt:
            start_episode = ckpt["episode"] + 1
            total_env_steps = ckpt["total_steps"]
            best_score = ckpt["best_score"]
            reward_history = ckpt["reward_history"]
            for r in reward_history[-50:]:
                recent_scores.append(r)
            print(f" Resumed from Episode {ckpt['episode']} | All-Time Best: {best_score:.1f}")

    interrupted = False
    paused = False
    fast_mode = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\n [INTERRUPT] Caught exit signal. Saving checkpoint before shutdown...")
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)

    print(f"{'Ep':>5} | {'Steps':>6} | {'Official Score':>14} | {'Avg (50)':>9} | {'Best':>8} | {'Official Result':<18}")
    print("-" * 75)

    step_in_rollout = 0
    ep = start_episode
    ep_score = 0.0

    try:
        for ep in range(start_episode, args.episodes + 1):
            if interrupted:
                break

            # Dynamic Learning Rate Annealing
            progress_frac = (ep - 1) / max(1, args.episodes - 1)
            current_lr = args.lr_start - progress_frac * (args.lr_start - args.lr_end)
            agent.set_learning_rate(max(args.lr_end, current_lr))

            state, _ = env.reset()
            ep_score = 0.0
            ep_steps = 0
            outcome = "IN FLIGHT"

            while True:
                #  Pygame Hotkeys
                if args.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            interrupted = True
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                paused = not paused
                                print(f" [SPACE] {'PAUSED' if paused else 'RESUMED'}", flush=True)
                            elif event.key == pygame.K_v:
                                fast_mode = not fast_mode
                                env.unwrapped.metadata["render_fps"] = 1000 if fast_mode else 50
                                print(f" [MODE] {'FAST TRAINING (Uncapped FPS)' if fast_mode else 'REALTIME WATCH (50 FPS)'}", flush=True)
                            elif event.key == pygame.K_s:
                                checkpoint_mgr.save_checkpoint(
                                    agent, ep, total_env_steps, ep_score,
                                    float(np.mean(recent_scores)) if recent_scores else 0.0,
                                    best_score, reward_history
                                )
                                print(" [HOTKEY-S] Checkpoint manually saved to disk!", flush=True)
                            elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                                interrupted = True
                                break

                    # Handle Pause state
                    while paused and not interrupted:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                                interrupted = True
                                break
                            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                                paused = False
                                print(" [SPACE] RESUMED", flush=True)
                        time.sleep(0.05)

                    if interrupted:
                        break

                action, log_prob, value = agent.select_action(state)
             
                env_action = np.clip(action, -1.0, 1.0)
                next_state, reward, terminated, truncated, info = env.step(env_action)

                done = terminated or truncated
                agent.buffer.add(state, action, log_prob, reward, done, value)

                state = next_state
                ep_score += reward
                ep_steps += 1
                total_env_steps += 1
                step_in_rollout += 1

                # Draw Sleek Telemetry HUD Overlay 
                if args.render and env.unwrapped.screen is not None:
                    try:
                        hud_bg = pygame.Surface((600, 38), pygame.SRCALPHA)
                        hud_bg.fill((10, 15, 25, 215))
                        pygame.draw.rect(hud_bg, (50, 150, 220), (0, 0, 600, 38), 1)

                        avg_sc = float(np.mean(recent_scores)) if recent_scores else 0.0
                        hud_txt = f"EP:{ep:03d} | SCORE:{ep_score:>5.1f} | AVG:{avg_sc:>5.1f} | BEST:{best_score:>5.1f} | {'FAST' if fast_mode else 'WATCH'}"
                        txt_surf = hud_font.render(hud_txt, True, (100, 220, 255))
                        hud_bg.blit(txt_surf, (10, 4))

                        k_txt = "[Space]:Pause  [V]:Fast/Watch  [S]:Save  [Q]:Quit"
                        k_surf = key_font.render(k_txt, True, (180, 190, 205))
                        hud_bg.blit(k_surf, (10, 20))

                        env.unwrapped.screen.blit(hud_bg, (0, 0))
                        pygame.display.flip()
                    except Exception:
                        pass

                # PPO Network Update
                if step_in_rollout >= args.update_steps:
                    next_val = agent.get_value(next_state) if not done else 0.0
                    agent.update(next_val)
                    step_in_rollout = 0

                if done:
                    if terminated:
                        if ep_score >= 200.0:
                            outcome = "SOLVED TOUCHDOWN!"
                        elif ep_score > 50.0:
                            outcome = "SOFT TOUCHDOWN"
                        else:
                            outcome = "CRASH / TILTED"
                    else:
                        outcome = "TIMEOUT"
                    break

            if interrupted:
                break

            reward_history.append(ep_score)
            recent_scores.append(ep_score)
            avg_score = float(np.mean(recent_scores))

            is_best = ep_score > best_score
            if is_best:
                best_score = ep_score

            print(f"{ep:>5d} | {ep_steps:>6d} | {ep_score:>14.1f} | {avg_score:>9.1f} | {best_score:>8.1f} | {outcome:<18}", flush=True)

            # Checkpoint 
            if is_best or ep % args.save_freq == 0:
                checkpoint_mgr.save_checkpoint(
                    agent, ep, total_env_steps, ep_score, avg_score, best_score, reward_history, is_best=is_best
                )

    finally:
        print("\n [SAVING] Ensuring final training state is preserved...", flush=True)
        checkpoint_mgr.save_checkpoint(
            agent, ep, total_env_steps, ep_score,
            float(np.mean(recent_scores)) if recent_scores else 0.0,
            best_score, reward_history
        )
        env.close()
        print(" [SHUTDOWN] LunarLander Training Session Complete. Progress saved.\n", flush=True)


if __name__ == "__main__":
    main()
