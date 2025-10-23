#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "900"
os.environ["DATASETS_HTTP_TIMEOUT"] = "900"
os.environ["HF_TOKEN"] = ""


DownloadItem = Tuple[str, Union[str, Iterable[str]]]


@dataclass(frozen=True)
class DownloadTarget:
    repo_id: str
    files: Tuple[str, ...]

    @classmethod
    def from_item(cls, item: DownloadItem) -> "DownloadTarget":
        repo_id, filenames = item
        if isinstance(filenames, str):
            files: Tuple[str, ...] = (filenames,)
        else:
            files = tuple(filenames)
        return cls(repo_id=repo_id, files=files)

    def iter_tasks(self) -> Iterable[Tuple[str, str]]:
        for fname in self.files:
            yield self.repo_id, fname


STATIC_FILES: List[DownloadItem] = [
    ("shoumenchougou/RWKV7-G0a2-7.2B-GGUF", "rwkv7-g0a2-7.2b-Q4_K_M.gguf"),
    ("shoumenchougou/RWKV7-G1a-2.9B-GGUF", "rwkv7-g1a-2.9b-Q6_K.gguf"),
    ("shoumenchougou/RWKV7-G1a3-1.5B-GGUF", "rwkv7-g1a3-1.5b-Q6_K.gguf"),
    # ("shoumenchougou/RWKV7-G1a-0.4B-GGUF", "rwkv7-g1a-0.4b-FP16.gguf"),
    # ("shoumenchougou/RWKV7-G1a-0.1B-GGUF", "rwkv7-g1a-0.1b-FP16.gguf"),
]

# 自动抓取 .pth 文件的模型仓库列表
PTH_REPOS: Tuple[str, ...] = ("BlinkDL/rwkv7-g1",)

DEFAULT_OUT_DIR = "/public/home/ssjxzkz/Weights"
DEFAULT_REVISION = "main"
MAX_AUTO_WORKERS = 8
INITIAL_BACKOFF_SECONDS = 5
MAX_BACKOFF_SECONDS = 300


def discover_pth_files(api: HfApi, repo_id: str, revision: str = DEFAULT_REVISION) -> Tuple[str, ...]:
    """列出仓库中的全部 .pth 文件。"""
    try:
        repo_files = api.list_repo_files(
            repo_id=repo_id,
            revision=revision,
            repo_type="model",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 无法获取 {repo_id} 的 .pth 列表：{exc}")
        return ()

    pth_files = tuple(sorted(fname for fname in repo_files if fname.endswith(".pth")))
    if not pth_files:
        print(f"⚠️  未在 {repo_id} 找到任何 .pth 文件")
        return ()

    print(f"🔍  {repo_id} 发现 {len(pth_files)} 个 .pth 文件")
    return pth_files


def build_download_targets(api: HfApi) -> List[DownloadTarget]:
    """组装静态和动态下载任务列表。"""
    targets: List[DownloadTarget] = [DownloadTarget.from_item(item) for item in STATIC_FILES]

    for repo_id in PTH_REPOS:
        pth_files = discover_pth_files(api, repo_id)
        if pth_files:
            targets.append(DownloadTarget(repo_id=repo_id, files=pth_files))

    return targets


def download_one(repo_id: str, filename: str, out_dir: Path) -> Path:
    """使用 hf_hub_download 下载单个文件，支持断点续传。"""
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=DEFAULT_REVISION,
        local_dir=str(out_dir / repo_id.replace("/", "__")),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return Path(local_path)


def download_with_retry(repo_id: str, filename: str, out_dir: Path) -> Path:
    """无限重试下载，出现错误指数退避再试。"""
    attempt = 1
    delay = INITIAL_BACKOFF_SECONDS

    while True:
        try:
            return download_one(repo_id, filename, out_dir)
        except KeyboardInterrupt:
            raise
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
            wait = min(delay, MAX_BACKOFF_SECONDS)
            print(f"⚠️  重试：{repo_id}/{filename} (第 {attempt} 次失败，{wait}s 后重试) | {exc}")
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001
            wait = min(delay, MAX_BACKOFF_SECONDS)
            print(
                f"⚠️  重试：{repo_id}/{filename} (第 {attempt} 次失败，{wait}s 后重试) | "
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(wait)

        attempt += 1
        delay = min(delay * 2, MAX_BACKOFF_SECONDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 Hugging Face 权重到本地。")
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=DEFAULT_OUT_DIR,
        help=f"保存目录（默认：{DEFAULT_OUT_DIR}）",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="并发线程数（默认：自动根据下载任务数与 CPU 核心数决定）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    targets = build_download_targets(api)

    if not targets:
        print("⚠️  没有任何下载任务，退出。")
        return

    total_tasks = sum(len(target.files) for target in targets)
    default_workers = min(
        max(1, (os.cpu_count() or 4) * 2),
        MAX_AUTO_WORKERS,
        total_tasks or 1,
    )
    max_workers = args.workers or default_workers

    print(f"保存目录：{out_dir}")
    print(f"总任务数：{total_tasks} | 并发线程数：{max_workers}")

    results: Dict[Tuple[str, str], Path] = {}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(download_with_retry, repo, fname, out_dir): (repo, fname)
                for target in targets
                for repo, fname in target.iter_tasks()
            }

            for future in as_completed(future_map):
                repo, fname = future_map[future]
                path = future.result()
                size_gb = path.stat().st_size / (1024**3)
                print(f"✅ 已下载：{repo}/{fname} -> {path}  ({size_gb:.2f} GB)")
                results[(repo, fname)] = path
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        sys.exit(130)

    print("\n=== 下载完成 ===")
    for target in targets:
        print(f"{target.repo_id}:")
        for fname in target.files:
            path = results.get((target.repo_id, fname))
            if path is None:
                print(f"  - {fname}: 未知状态（可能被中断）")
            else:
                print(f"  - {fname}: {path}")


if __name__ == "__main__":
    # 可选：在国内网络环境可配置镜像或代理提升稳定性
    # 也可以设置环境变量 HF_TOKEN 使用你的 Hugging Face 访问令牌（若仓库私有）
    main()
