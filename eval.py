import os
import time
import argparse
import numpy as np
import pygame
import gymnasium as gym

from agent import PPOAgent
from checkpoint_manager import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Official LunarLanderContinuous-v3 Benchmark Evaluation")
    parser.add_argument("--episodes", type=int, default=100, help="Number of benchmark evaluation missions (default 100)")
    parser.add_argument("--render", action="store_true", default=False, help="Render visually")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("  OFFICIAL BENCHMARK EVALUATION: LUNARLANDERCONTINUOUS-V3")
    print("=" * 75)

    if not os.path.exists(args.checkpoint):
        if os.path.exists("checkpoints/latest_checkpoint.pth"):
            args.checkpoint = "checkpoints/latest_checkpoint.pth"
        else:
            print(f" [ERROR] No checkpoint found at '{args.checkpoint}'. Run train.py first!")
            return

    render_mode = "human" if args.render else None
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        enable_wind=True,
        wind_power=20.0,
        turbulence_power=2.0,
        render_mode=render_mode,
    )

    agent = PPOAgent(state_dim=8, action_dim=2)
    checkpoint_mgr = CheckpointManager()

    ckpt = checkpoint_mgr.load_checkpoint(
        agent,
        load_best=("best_model" in args.checkpoint),
        specific_path=args.checkpoint,
    )
    if ckpt is None:
        print(" Failed to load checkpoint weights.")
        return

    print(f" [MODEL] Loaded from: {args.checkpoint} (Trained for {ckpt['episode']} episodes)")
    print(f" [BENCHMARK] Running {args.episodes} consecutive standardized missions under max turbulence...")
    print(f" [TARGET] Mean score >= +200.0 points")
    if args.render:
        print(" [KEYS] [Space]=Pause/Resume  [V]=Toggle Fast/Watch  [Q]=Quit\n")
        pygame.font.init()
        hud_font = pygame.font.SysFont("Consolas", 13, bold=True)
        key_font = pygame.font.SysFont("Consolas", 12)
    else:
        print()

    scores = []
    landings = 0
    crashes = 0
    timeouts = 0

    paused = False
    fast_mode = False
    interrupted = False

    for ep in range(1, args.episodes + 1):
        if interrupted:
            break

        state, _ = env.reset(seed=ep * 42)
        ep_score = 0.0

        while True:
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
                            print(f" [MODE] {'FAST SPEED (Uncapped FPS)' if fast_mode else 'REALTIME WATCH (50 FPS)'}", flush=True)
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            interrupted = True
                            break

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

            action = agent.select_deterministic_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state
            ep_score += reward
            if args.render and env.unwrapped.screen is not None:
                try:
                    hud_bg = pygame.Surface((600, 38), pygame.SRCALPHA)
                    hud_bg.fill((10, 15, 25, 215))
                    pygame.draw.rect(hud_bg, (50, 150, 220), (0, 0, 600, 38), 1)

                    curr_mean = float(np.mean(scores)) if scores else 0.0
                    hud_txt = f"EP:{ep:03d}/{args.episodes} | SCORE:{ep_score:>5.1f} | MEAN:{curr_mean:>5.1f} | WIN:{landings} | {'FAST' if fast_mode else 'WATCH'}"
                    txt_surf = hud_font.render(hud_txt, True, (100, 220, 255))
                    hud_bg.blit(txt_surf, (10, 4))

                    k_txt = "[Space]:Pause  [V]:Fast/Watch  [Q]:Quit"
                    k_surf = key_font.render(k_txt, True, (180, 190, 205))
                    hud_bg.blit(k_surf, (10, 20))

                    env.unwrapped.screen.blit(hud_bg, (0, 0))
                    pygame.display.flip()
                except Exception:
                    pass

            if terminated or truncated:
                if ep_score >= 200.0:
                    outcome = "SOLVED LANDING (+200)"
                    landings += 1
                elif terminated and ep_score > 50.0:
                    outcome = "SOFT TOUCHDOWN"
                    landings += 1
                elif terminated:
                    outcome = "CRASH / TILTED"
                    crashes += 1
                else:
                    outcome = "TIMEOUT"
                    timeouts += 1

                scores.append(ep_score)
                if ep % 10 == 0 or ep <= 10:
                    print(f" Mission #{ep:03d} | Official Score: {ep_score:>7.1f} | Result: {outcome}")
                break

    env.close()

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    success_rate = (landings / args.episodes) * 100.0

    print("\n" + "=" * 75)
    print("  OFFICIAL GYMNASIUM BENCHMARK REPORT")
    print("=" * 75)
    print(f" Total Evaluation Episodes:    {args.episodes}")
    print(f" Mean Official Score:          {mean_score:.2f} +/- {std_score:.2f}")
    print(f" Successful Landings:          {landings} / {args.episodes} ({success_rate:.1f}%)")
    print(f" Crashes:                      {crashes} / {args.episodes}")
    print(f" Timeouts:                     {timeouts} / {args.episodes}")
    print("-" * 75)

    if mean_score >= 200.0:
        print(" [RESULT] STATUS: OFFICIALLY SOLVED! (Score >= +200.0 Benchmark Standard)")
    else:
        print(f" [RESULT] STATUS: IN PROGRESS ({mean_score:.1f} / 200.0 target). Train for more episodes.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

