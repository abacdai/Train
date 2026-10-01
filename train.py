from __future__ import annotations

import argparse
import copy
import os
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from books import Book
from brain import (
    MCTS, PolicyValueNet, choose_action, train_network
)
from engine import State, apply

RULE_VERSION = "5x5-3start-fifo-NSWE-immediate-after-explosion-v1"

def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)

def load_local_checkpoint(path, device="cpu"):
    # Chỉ mở checkpoint tự tạo hoặc từ nguồn bạn tin cậy.
    # weights_only=False có thể thực thi pickle từ tệp độc hại.
    return torch.load(
        path, map_location=device, weights_only=False
    )

def cpu_weights(model):
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }

def wilson_lower(wins, games, z=1.96):
    if games == 0:
        return 0.0
    p = wins / games
    denominator = 1 + z * z / games
    center = p + z * z / (2 * games)
    radius = z * np.sqrt(
        p * (1 - p) / games + z * z / (4 * games * games)
    )
    return float((center - radius) / denominator)

def model_move(model, state, rng, simulations, stochastic=False):
    policy = MCTS(
        model, simulations=simulations, rng=rng
    ).search(state, noise=False)

    return choose_action(
        policy, rng, temperature=1.0 if stochastic else 0.0
    )

def arena(candidate, incumbent, rng, games, simulations):
    wins = 0

    for game in range(games):
        candidate_side = 1 if game % 2 == 0 else -1
        state = State()
        ply = 0

        while not state.winner:
            model = (
                candidate if state.turn == candidate_side
                else incumbent
            )
            action = model_move(
                model, state, rng, simulations,
                stochastic=ply < 2,
            )
            state = apply(state, action)
            ply += 1

        wins += int(state.winner == candidate_side)

    return wins

def random_baseline(model, rng, games, simulations):
    wins = 0

    for game in range(games):
        model_side = 1 if game % 2 == 0 else -1
        state = State()
        ply = 0

        while not state.winner:
            if state.turn == model_side:
                action = model_move(
                    model, state, rng, simulations,
                    stochastic=ply < 2,
                )
            else:
                action = int(rng.choice(state.legal_actions()))

            state = apply(state, action)
            ply += 1

        wins += int(state.winner == model_side)

    return wins

def self_play(best, args, rng, root, completed_games):
    pending_path = root / "pending.pt"
    pending = None

    if pending_path.exists():
        loaded = load_local_checkpoint(pending_path)
        if (
            loaded["completed_games"] == completed_games
            and loaded["rules"] == RULE_VERSION
        ):
            pending = loaded
            rng.bit_generator.state = pending["rng"]
            print(
                f"Tiếp tục ván {pending['id']}, "
                f"đã có {len(pending['records'])} nước."
            )

    if pending is None:
        pending = {
            "id": uuid.uuid4().hex,
            "rules": RULE_VERSION,
            "completed_games": completed_games,
            "state": State().to_dict(),
            "records": [],
            "rng": copy.deepcopy(rng.bit_generator.state),
        }
        atomic_save(pending, pending_path)

    state = State.from_dict(pending["state"])

    while not state.winner:
        policy = MCTS(
            best, simulations=args.simulations, rng=rng
        ).search(state, noise=True)

        ply = len(pending["records"])
        temperature = 1.0 if ply < args.explore_plies else 0.25
        action = choose_action(policy, rng, temperature)

        pending["records"].append({
            "state": state.to_dict(),
            "policy": policy,
            "action": action,
        })

        state = apply(state, action)
        pending["state"] = state.to_dict()
        pending["rng"] = copy.deepcopy(rng.bit_generator.state)

        # Không lưu toàn bộ replay mỗi nước.
        # Chỉ lưu ván đang diễn ra để phục hồi sau ngắt phiên.
        atomic_save(pending, pending_path)

    return pending, state.winner

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/chain")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)

    # Tổng số chu kỳ mong muốn, không phải số chu kỳ bổ sung.
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iteration", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--eval-simulations", type=int, default=128)
    parser.add_argument("--explore-plies", type=int, default=10)

    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)

    parser.add_argument("--arena-games", type=int, default=40)
    parser.add_argument("--baseline-games", type=int, default=10)

    parser.add_argument(
        "--wandb-project", default=None,
        help="Chỉ gửi dữ liệu W&B khi bạn chủ động đặt tùy chọn này.",
    )
    return parser

def main():
    args = build_parser().parse_args()

    positive = (
        "channels", "blocks", "iterations",
        "games_per_iteration", "simulations", "eval_simulations",
        "train_steps", "batch_size", "replay_size", "arena_games",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} phải >= 1.")

    if args.arena_games < 2 or args.arena_games % 2:
        raise ValueError("arena-games phải là số chẵn >= 2.")
    if args.baseline_games < 0 or args.baseline_games % 2:
        raise ValueError("baseline-games phải là số chẵn >= 0.")
    if args.explore_plies < 0 or args.learning_rate <= 0:
        raise ValueError("explore-plies hoặc learning-rate không hợp lệ.")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Không có GPU CUDA khả dụng.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    config = {
        "channels": args.channels,
        "blocks": args.blocks,
        "rules": RULE_VERSION,
    }

    best = PolicyValueNet(
        channels=args.channels, blocks=args.blocks
    ).to(device)
    best.eval()

    replay = deque(maxlen=args.replay_size)
    progress = {
        "iteration": 0,
        "completed_games": 0,
        "since_update": 0,
        "champion_version": 0,
    }

    checkpoint_path = root / "trainer.pt"

    if checkpoint_path.exists():
        checkpoint = load_local_checkpoint(checkpoint_path)
        if checkpoint["config"] != config:
            raise ValueError(
                "Kiến trúc hoặc luật khác checkpoint. "
                "Dùng thư mục --out khác cho thí nghiệm mới."
            )

        best.load_state_dict(checkpoint["best"])
        replay.extend(checkpoint["replay"])
        progress = checkpoint["progress"]
        rng.bit_generator.state = checkpoint["rng"]
        torch.set_rng_state(checkpoint["torch_rng"])

        cuda_rng = checkpoint.get("cuda_rng")
        if (
            cuda_rng is not None
            and torch.cuda.is_available()
            and len(cuda_rng) == torch.cuda.device_count()
        ):
            torch.cuda.set_rng_state_all(cuda_rng)

    def save_training():
        atomic_save({
            "config": config,
            "best": cpu_weights(best),
            "replay": list(replay),
            "progress": dict(progress),
            "rng": copy.deepcopy(rng.bit_generator.state),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
        }, checkpoint_path)

    def save_champion():
        atomic_save({
            "config": config,
            "weights": cpu_weights(best),
            "champion_version": progress["champion_version"],
        }, root / "best.pt")

    # trainer.pt là nguồn sự thật; tạo lại best.pt khi resume.
    save_training()
    save_champion()

    writer = SummaryWriter(str(root / "tensorboard"))
    book = Book(root / "books.sqlite")
    wandb_run = None

    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )

    print(f"Thiết bị: {device}")
    print(f"Chu kỳ đã hoàn tất: {progress['iteration']}")

    try:
        while progress["iteration"] < args.iterations:
            while progress["since_update"] < args.games_per_iteration:
                pending, winner = self_play(
                    best, args, rng, root,
                    progress["completed_games"],
                )

                records = pending["records"]

                # Idempotent: không đếm lại ván đã ghi trong SQLite.
                book.record_game(pending["id"], records, winner)

                for record in records:
                    state = State.from_dict(record["state"])
                    outcome = 1.0 if state.turn == winner else -1.0
                    replay.append((
                        state.encode(),
                        state.mask(),
                        np.asarray(record["policy"], dtype=np.float32),
                        outcome,
                    ))

                progress["completed_games"] += 1
                progress["since_update"] += 1

                game_number = progress["completed_games"]
                writer.add_scalar(
                    "selfplay/plies", len(records), game_number
                )
                writer.add_scalar(
                    "selfplay/p1_win", int(winner == 1), game_number
                )

                # Ghi checkpoint trước khi xóa ván pending.
                save_training()
                (root / "pending.pt").unlink(missing_ok=True)

                print(
                    f"Ván {game_number}: "
                    f"{len(records)} nước, "
                    f"{'P1' if winner == 1 else 'P2'} thắng."
                )

            candidate = copy.deepcopy(best)
            metrics = train_network(
                candidate,
                list(replay),
                rng,
                steps=args.train_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )

            wins = arena(
                candidate, best, rng,
                args.arena_games, args.eval_simulations,
            )
            lower = wilson_lower(wins, args.arena_games)
            promoted = lower > 0.5

            if promoted:
                best.load_state_dict(candidate.state_dict())
                progress["champion_version"] += 1

            progress["iteration"] += 1
            progress["since_update"] = 0
            iteration = progress["iteration"]

            logs = {
                **{f"train/{k}": v for k, v in metrics.items()},
                "arena/candidate_win_rate": wins / args.arena_games,
                "arena/wilson_lower_95": lower,
                "arena/promoted": int(promoted),
                "data/replay_size": len(replay),
                "model/champion_version": progress["champion_version"],
            }

            if args.baseline_games:
                baseline_wins = random_baseline(
                    best, rng,
                    args.baseline_games, args.eval_simulations,
                )
                logs["baseline/win_rate_vs_random"] = (
                    baseline_wins / args.baseline_games
                )

            save_training()
            save_champion()
            book.export(root / "textbook")

            for key, value in logs.items():
                writer.add_scalar(key, value, iteration)
            writer.flush()

            if wandb_run is not None:
                wandb_run.log(logs, step=iteration)

            print(
                f"Chu kỳ {iteration}: "
                f"ứng viên thắng {wins}/{args.arena_games}; "
                f"Wilson thấp={lower:.3f}; "
                f"{'NHẬN mô hình mới' if promoted else 'GIỮ mô hình cũ'}"
            )

    except KeyboardInterrupt:
        print(
            "\nĐã ngắt. Tiến trình đã ghi trên đĩa được giữ nguyên. "
            "Chạy lại cùng lệnh để tiếp tục."
        )
        # Không ghi đè checkpoint ở đây:
        # có thể đang giữa bước tích hợp một ván hoặc huấn luyện.
    finally:
        writer.close()
        book.close()
        if wandb_run is not None:
            wandb_run.finish()

if __name__ == "__main__":
    main()
