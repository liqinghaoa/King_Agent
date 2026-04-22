#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Parse Tencent Kaiwu local training logs into same-definition Markdown metrics.

Usage:
    python scripts/parse_kaiwu_training_log.py --run diy=path/to/log_or_dir
    python scripts/parse_kaiwu_training_log.py --run ppo=ppo_logs --run diy=diy_logs
"""

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

METRIC_ALIASES = {
    "total_score": ["total_score", "sim_score", "score"],
    "finished_steps": ["finished_steps", "episode_steps", "steps"],
    "step_score": ["step_score"],
    "treasure_score": ["treasure_score"],
    "treasures_collected": ["treasures_collected", "treasure_collected_count"],
    "reward": ["reward", "total_reward"],
    "total_loss": ["total_loss"],
    "value_loss": ["value_loss"],
    "policy_loss": ["policy_loss"],
    "entropy_loss": ["entropy_loss", "entropy"],
    "succ_cnt": ["succ_cnt", "success_cnt"],
    "flash_count": ["flash_count"],
    "flash_rate": ["flash_rate"],
    "effective_flash_count": ["effective_flash_count"],
    "ineffective_flash_count": ["ineffective_flash_count"],
    "effective_flash_rate": ["effective_flash_rate"],
    "ineffective_flash_rate": ["ineffective_flash_rate"],
    "danger_flash_count": ["danger_flash_count"],
    "safe_flash_count": ["safe_flash_count"],
    "unknown_flash_count": ["unknown_flash_count"],
    "danger_flash_rate": ["danger_flash_rate"],
    "safe_flash_rate": ["safe_flash_rate"],
    "danger_effective_flash_count": ["danger_effective_flash_count"],
    "danger_ineffective_flash_count": ["danger_ineffective_flash_count"],
    "danger_effective_flash_rate": ["danger_effective_flash_rate"],
    "danger_ineffective_flash_rate": ["danger_ineffective_flash_rate"],
    "escape_flash_count": ["escape_flash_count"],
    "invalid_flash_count": ["invalid_flash_count"],
    "flash_reward_sum": ["flash_reward_sum"],
    "flash_distance_delta_sum": ["flash_distance_delta_sum"],
    "avg_flash_distance_delta_per_flash": ["avg_flash_distance_delta_per_flash"],
    "avg_flash_reward_per_flash": ["avg_flash_reward_per_flash"],
    "flash_survive_5_rate": ["flash_survive_5_rate"],
    "post_flash_survive_5_rate": ["post_flash_survive_5_rate"],
    "flash_eval_trigger_count": ["flash_eval_trigger_count"],
    "flash_execute_count": ["flash_execute_count"],
    "flash_execute_rate": ["flash_execute_rate"],
    "flash_skip_cooldown_count": ["flash_skip_cooldown_count"],
    "flash_skip_no_safe_candidate_count": ["flash_skip_no_safe_candidate_count"],
    "flash_skip_move_better_count": ["flash_skip_move_better_count"],
    "best_flash_better_than_move_count": ["best_flash_better_than_move_count"],
    "best_move_better_than_flash_count": ["best_move_better_than_flash_count"],
    "flash_leave_threat_count": ["flash_leave_threat_count"],
    "flash_leave_threat_rate": ["flash_leave_threat_rate"],
    "avg_flash_distance_gain": ["avg_flash_distance_gain"],
    "avg_flash_min_margin_gain": ["avg_flash_min_margin_gain"],
    "avg_flash_openness_gain": ["avg_flash_openness_gain"],
    "invalid_flash_rate": ["invalid_flash_rate"],
    "post_flash_dead_end_count": ["post_flash_dead_end_count"],
    "post_flash_dead_end_rate": ["post_flash_dead_end_rate"],
    "early_flash_count": ["early_flash_count"],
    "early_flash_episode_count": ["early_flash_episode_count"],
    "wall_cross_flash_count": ["wall_cross_flash_count"],
    "wall_cross_effective_count": ["wall_cross_effective_count"],
    "wall_cross_effective_rate": ["wall_cross_effective_rate"],
    "choke_escape_flash_count": ["choke_escape_flash_count"],
    "choke_escape_effective_count": ["choke_escape_effective_count"],
    "choke_escape_effective_rate": ["choke_escape_effective_rate"],
}

COUNTER_ALIASES = {
    "train_global_step": ["train_global_step", "train_count", "learner_train_count", "train_step"],
    "episode_cnt": ["episode_cnt", "episode"],
    "error_cnt": ["error_cnt"],
}

RESULT_RE = re.compile(r"\bresult\s*[:=]\s*([A-Za-z_]+)", re.IGNORECASE)
GAMEOVER_RE = re.compile(r"\bGAMEOVER\b", re.IGNORECASE)
KEY_VALUE_RE = re.compile(rf"\b([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*({NUM_RE})")
KEY_IS_RE = re.compile(rf"\b([A-Za-z_][A-Za-z0-9_]*)\s+is\s+({NUM_RE})", re.IGNORECASE)
TIMESTAMP_PATTERNS = [
    (re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)"), True),
    (re.compile(r"(\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)"), False),
]


@dataclass
class RunStats:
    label: str
    sources: List[Path] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, List[float]] = field(default_factory=lambda: {k: [] for k in METRIC_ALIASES})
    counters: Dict[str, List[float]] = field(default_factory=lambda: {k: [] for k in COUNTER_ALIASES})
    gameover_count: int = 0
    fail_count: int = 0
    success_count: int = 0
    total_lines: int = 0


def collect_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".log", ".txt", ".out", ".err"}
            )
        elif path.is_file():
            files.append(path)
    return sorted(set(files))


def read_lines(path: Path) -> Iterable[str]:
    for encoding in ("utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, errors="replace") as f:
                yield from f
            return
        except UnicodeError:
            continue


def parse_timestamp(line: str, default_year: int) -> Optional[datetime]:
    for pattern, has_year in TIMESTAMP_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        text = match.group(1).replace("/", "-").replace(",", ".").replace("T", " ")
        formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]
        if not has_year:
            text = f"{default_year}-{text}"
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
    return None


def literal_dicts(line: str) -> Iterable[dict]:
    for start in [m.start() for m in re.finditer(r"\{", line)]:
        end_offsets = [m.end() for m in re.finditer(r"\}", line[start:])]
        for end_offset in reversed(end_offsets):
            chunk = line[start : start + end_offset].strip()
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(chunk)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    yield parsed
                break
            else:
                continue
            break
        else:
            continue


def flatten_dict(data: dict, prefix: str = "") -> Iterable[Tuple[str, object]]:
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from flatten_dict(value, name)
        else:
            yield name, value


def to_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and re.fullmatch(NUM_RE, value.strip()):
        return float(value)
    return None


def update_time(stats: RunStats, line: str, default_year: int) -> None:
    ts = parse_timestamp(line, default_year)
    if ts is None:
        return
    if stats.start_time is None or ts < stats.start_time:
        stats.start_time = ts
    if stats.end_time is None or ts > stats.end_time:
        stats.end_time = ts


def add_key_values(stats: RunStats, key_values: Iterable[Tuple[str, object]]) -> None:
    normalized: Dict[str, float] = {}
    for raw_key, raw_value in key_values:
        key = str(raw_key).split(".")[-1]
        value = to_float(raw_value)
        if value is None:
            continue
        normalized[key] = value

    for canonical, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                stats.metrics[canonical].append(normalized[alias])
                break

    for canonical, aliases in COUNTER_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                stats.counters[canonical].append(normalized[alias])
                break


def parse_result(stats: RunStats, line: str) -> None:
    if GAMEOVER_RE.search(line):
        stats.gameover_count += 1

    match = RESULT_RE.search(line)
    if not match:
        return
    result = match.group(1).upper()
    if result in {"FAIL", "FAILED", "LOSE", "LOSS", "TERMINATED"}:
        stats.fail_count += 1
    elif result in {"SUCCESS", "SUCC", "WIN", "COMPLETED", "TRUNCATED"}:
        stats.success_count += 1


def parse_run(label: str, paths: List[str], default_year: int) -> RunStats:
    stats = RunStats(label=label)
    files = collect_files(paths)
    stats.sources = files

    for path in files:
        for line in read_lines(path):
            stats.total_lines += 1
            update_time(stats, line, default_year)
            parse_result(stats, line)
            add_key_values(stats, KEY_VALUE_RE.findall(line))
            add_key_values(stats, KEY_IS_RE.findall(line))
            for data in literal_dicts(line):
                add_key_values(stats, flatten_dict(data))

    return stats


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def fmt_dt(value: Optional[datetime]) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else "-"


def metric_summary(values: List[float]) -> Tuple[str, str, str, str, str]:
    if not values:
        return "-", "-", "-", "-", "0"
    return (
        fmt(sum(values)),
        fmt(sum(values) / len(values)),
        fmt(max(values)),
        fmt(values[-1]),
        str(len(values)),
    )


def duration_text(stats: RunStats) -> str:
    if stats.start_time is None or stats.end_time is None:
        return "-"
    seconds = int((stats.end_time - stats.start_time).total_seconds())
    if seconds < 0:
        return "-"
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def latest_or_max(values: List[float]) -> Tuple[str, str]:
    if not values:
        return "-", "-"
    return fmt(max(values)), fmt(values[-1])


def render_markdown(runs: List[RunStats]) -> str:
    lines: List[str] = []
    lines.append("# Kaiwu Training Log Summary")
    lines.append("")
    lines.append("## Run Overview")
    lines.append("| run | files | lines | start | end | duration | train_global_step/latest | episode_cnt/latest | GAMEOVER | FAIL | SUCCESS | error_cnt/sum |")
    lines.append("| :-- | --: | --: | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |")
    for stats in runs:
        train_max, train_latest = latest_or_max(stats.counters["train_global_step"])
        episode_max, episode_latest = latest_or_max(stats.counters["episode_cnt"])
        error_sum = sum(stats.counters["error_cnt"]) if stats.counters["error_cnt"] else 0
        lines.append(
            f"| {stats.label} | {len(stats.sources)} | {stats.total_lines} | {fmt_dt(stats.start_time)} | "
            f"{fmt_dt(stats.end_time)} | {duration_text(stats)} | {train_latest or train_max} | "
            f"{episode_latest or episode_max} | {stats.gameover_count} | {stats.fail_count} | "
            f"{stats.success_count} | {fmt(error_sum)} |"
        )

    lines.append("")
    lines.append("## Metrics")
    lines.append("| run | metric | sum | avg | max | latest | count |")
    lines.append("| :-- | :-- | --: | --: | --: | --: | --: |")
    metric_order = [
        "total_score",
        "finished_steps",
        "step_score",
        "treasure_score",
        "treasures_collected",
        "reward",
        "total_loss",
        "value_loss",
        "policy_loss",
        "entropy_loss",
        "succ_cnt",
        "flash_count",
        "flash_rate",
        "effective_flash_count",
        "ineffective_flash_count",
        "effective_flash_rate",
        "ineffective_flash_rate",
        "danger_flash_count",
        "safe_flash_count",
        "unknown_flash_count",
        "danger_flash_rate",
        "safe_flash_rate",
        "danger_effective_flash_count",
        "danger_ineffective_flash_count",
        "danger_effective_flash_rate",
        "danger_ineffective_flash_rate",
        "escape_flash_count",
        "invalid_flash_count",
        "flash_reward_sum",
        "flash_distance_delta_sum",
        "avg_flash_distance_delta_per_flash",
        "avg_flash_reward_per_flash",
        "flash_survive_5_rate",
        "post_flash_survive_5_rate",
        "flash_eval_trigger_count",
        "flash_execute_count",
        "flash_execute_rate",
        "flash_skip_cooldown_count",
        "flash_skip_no_safe_candidate_count",
        "flash_skip_move_better_count",
        "best_flash_better_than_move_count",
        "best_move_better_than_flash_count",
        "flash_leave_threat_count",
        "flash_leave_threat_rate",
        "avg_flash_distance_gain",
        "avg_flash_min_margin_gain",
        "avg_flash_openness_gain",
        "invalid_flash_rate",
        "post_flash_dead_end_count",
        "post_flash_dead_end_rate",
        "early_flash_count",
        "early_flash_episode_count",
        "wall_cross_flash_count",
        "wall_cross_effective_count",
        "wall_cross_effective_rate",
        "choke_escape_flash_count",
        "choke_escape_effective_count",
        "choke_escape_effective_rate",
    ]
    for stats in runs:
        for metric in metric_order:
            total, avg, max_value, latest, count = metric_summary(stats.metrics[metric])
            lines.append(f"| {stats.label} | {metric} | {total} | {avg} | {max_value} | {latest} | {count} |")

    lines.append("")
    lines.append("## Derived Flash Rates")
    lines.append(
        "| run | flash_count | effective_flash_rate | ineffective_flash_rate | "
        "danger_flash_rate | safe_flash_rate | danger_effective_flash_rate | "
        "danger_ineffective_flash_rate | avg_flash_distance_delta_per_flash | avg_flash_reward_per_flash |"
    )
    lines.append("| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
    for stats in runs:
        flash_sum = sum(stats.metrics["flash_count"])
        effective_sum = sum(stats.metrics["effective_flash_count"])
        ineffective_sum = sum(stats.metrics["ineffective_flash_count"])
        danger_sum = sum(stats.metrics["danger_flash_count"])
        safe_sum = sum(stats.metrics["safe_flash_count"])
        danger_effective_sum = sum(stats.metrics["danger_effective_flash_count"])
        danger_ineffective_sum = sum(stats.metrics["danger_ineffective_flash_count"])
        flash_delta_sum = sum(stats.metrics["flash_distance_delta_sum"])
        flash_reward_sum = sum(stats.metrics["flash_reward_sum"])
        lines.append(
            f"| {stats.label} | {fmt(flash_sum)} | {fmt(effective_sum / flash_sum if flash_sum else 0)} | "
            f"{fmt(ineffective_sum / flash_sum if flash_sum else 0)} | "
            f"{fmt(danger_sum / flash_sum if flash_sum else 0)} | "
            f"{fmt(safe_sum / flash_sum if flash_sum else 0)} | "
            f"{fmt(danger_effective_sum / danger_sum if danger_sum else 0)} | "
            f"{fmt(danger_ineffective_sum / danger_sum if danger_sum else 0)} | "
            f"{fmt(flash_delta_sum / flash_sum if flash_sum else 0)} | "
            f"{fmt(flash_reward_sum / flash_sum if flash_sum else 0)} |"
        )

    lines.append("")
    lines.append("## Missing Fields")
    lines.append("| run | missing metrics | notes |")
    lines.append("| :-- | :-- | :-- |")
    for stats in runs:
        missing = [key for key in metric_order if not stats.metrics[key]]
        counter_missing = [key for key in COUNTER_ALIASES if not stats.counters[key]]
        notes = []
        if stats.gameover_count == 0:
            notes.append("no GAMEOVER lines")
        if not stats.sources:
            notes.append("no readable log files")
        all_missing = ", ".join(missing + counter_missing) if missing or counter_missing else "-"
        lines.append(f"| {stats.label} | {all_missing} | {', '.join(notes) if notes else '-'} |")

    lines.append("")
    lines.append("Field aliases: total_score also accepts sim_score/score; finished_steps also accepts episode_steps/steps; entropy_loss also accepts entropy; reward also accepts total_reward.")
    return "\n".join(lines)


def parse_run_args(run_args: List[str]) -> List[Tuple[str, List[str]]]:
    runs: List[Tuple[str, List[str]]] = []
    for item in run_args:
        if "=" not in item:
            raise SystemExit(f"--run must be label=path[,path...], got: {item}")
        label, paths_text = item.split("=", 1)
        paths = [p for p in paths_text.split(",") if p]
        runs.append((label.strip(), paths))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse Kaiwu training logs and output Markdown tables.")
    parser.add_argument("--run", action="append", required=True, help="Run input as label=path[,path...]")
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Default year for MM-DD timestamps")
    parser.add_argument("--output", help="Optional Markdown output path")
    args = parser.parse_args()

    runs = [parse_run(label, paths, args.year) for label, paths in parse_run_args(args.run)]
    markdown = render_markdown(runs)

    if args.output:
        Path(args.output).write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
