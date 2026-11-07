import os
import shutil
import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")

    def save_checkpoint(
        self,
        agent,
        episode,
        total_steps,
        score,
        avg_score,
        best_score,
        reward_history,
        is_best=False,
    ):
        checkpoint = {
            "episode": int(episode),
            "total_steps": int(total_steps),
            "score": float(score),
            "avg_score": float(avg_score),
            "best_score": float(best_score),
            "reward_history": [float(r) for r in reward_history],
            "network_state_dict": agent.network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        }

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        tmp_path = self.latest_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        shutil.move(tmp_path, self.latest_path)

        if is_best:
            best_tmp = self.best_path + ".tmp"
            torch.save(checkpoint, best_tmp)
            shutil.move(best_tmp, self.best_path)
            print(f" [RECORD] New all-time record score: {score:.1f}! Saved to {self.best_path}")

    def load_checkpoint(self, agent, load_best=False, specific_path=None):
        if specific_path and os.path.exists(specific_path):
            target_path = specific_path
        else:
            target_path = self.best_path if load_best else self.latest_path

        if not os.path.exists(target_path):
            if os.path.exists(self.latest_path):
                target_path = self.latest_path
            else:
                return None

        print(f" [CHECKPOINT] Resuming training from {target_path}...")
        checkpoint = torch.load(target_path, map_location=agent.device, weights_only=False)

        agent.network.load_state_dict(checkpoint["network_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "episode": checkpoint.get("episode", 0),
            "total_steps": checkpoint.get("total_steps", 0),
            "score": checkpoint.get("score", 0.0),
            "avg_score": checkpoint.get("avg_score", 0.0),
            "best_score": checkpoint.get("best_score", -9999.0),
            "reward_history": checkpoint.get("reward_history", []),
        }

    def has_checkpoint(self):
        return os.path.exists(self.latest_path) or os.path.exists(self.best_path)

