from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from typing import Callable, Any
from collections import Counter

try:
    from wcwidth import wcswidth
except Exception:
    wcswidth = None

try:
    import joblib
except Exception:
    joblib = None

try:
    import pandas as pd
except Exception:
    pd = None


# ---------------------------
# ANSI styling
# ---------------------------

def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


class C:
    if supports_color():
        RESET = "\033[0m"
        BOLD = "\033[1m"
        CYAN = "\033[96m"
        MAGENTA = "\033[95m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        DIM = "\033[2m"
        WHITE = "\033[97m"                 
        BG_UBUNTU = "\033[48;2;48;10;36m" 
    else:
        RESET = BOLD = CYAN = MAGENTA = GREEN = YELLOW = RED = BLUE = DIM = WHITE = BG_UBUNTU = ""


BG_UBUNTU = "\033[48;2;48;10;36m" if supports_color() else ""


def theme_wrap(line: str, theme: str) -> str:
    if theme == "aubergine":
        return f"{BG_UBUNTU}{line}{C.RESET}"
    return line

BIG_LOGO_PLAIN = """
███▄   ██    ███████    ██      ██    ██████
████▄  ██    ██         ██      ██    ██
██ ██▄ ██    █████      ██  ██  ██    ██████
██  █████    ██         ██ ████ ██        ██
██   ████    ███████     ███  ███     ██████

▄▀█  █ █  ▀█▀  █ █  █▀▀  █▄ █  ▀█▀  █  █▀▀  ▄▀█  ▀█▀  █▀█  █▀█
█▀█  █▄█   █   █▀█  ██▄  █ ▀█   █   █  █▄▄  █▀█   █   █▄█  █▀▄
""".strip("\n")


def terminal_width() -> int:
    cols = shutil.get_terminal_size((110, 40)).columns
    # Keep the launcher slightly narrower than the terminal so borders do not clip.
    return max(76, min(cols - 6, 100))


def _safe_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _query_nvidia_smi() -> tuple[str | None, str | None]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=1.0).strip()
        if not raw:
            return None, None
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) >= 4:
            util = parts[0]
            mem_used = float(parts[1])
            mem_total = float(parts[2])
            temp = parts[3]
            gpu_line = f"GPU {util}% | VRAM {mem_used/1024:.1f}/{mem_total/1024:.1f}G"
            temp_line = f"Temp {temp} °C"
            return gpu_line, temp_line
    except Exception:
        return None, None
    return None, None


def get_status_lines() -> list[str]:
    psutil = _safe_import_psutil()
    cpu_line = "CPU N/A | RAM N/A"
    if psutil is not None:
        try:
            cpu = psutil.cpu_percent(interval=0.0)
            ram = psutil.virtual_memory().percent
            cpu_line = f"CPU {cpu:.0f}% | RAM {ram:.0f}%"
        except Exception:
            pass

    gpu_line, temp_line = _query_nvidia_smi()
    if gpu_line is None:
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                used_gb = (total_bytes - free_bytes) / (1024 ** 3)
                total_gb = total_bytes / (1024 ** 3)
                gpu_line = f"GPU cuda | VRAM {used_gb:.1f}/{total_gb:.1f}G"
            else:
                gpu_line = f"GPU {current_device_label()}"
        except Exception:
            gpu_line = f"GPU {current_device_label()}"
    if temp_line is None:
        temp_line = "Temp N/A"

    return ["Status", f"Device Type: {current_device_label()}", cpu_line, gpu_line, temp_line]

def current_device_label() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def visual_width(text: str) -> int:
    text = str(text)
    if wcswidth is not None:
        value = wcswidth(text)
        return value if value >= 0 else len(text)
    return len(text)


def trim_to_width(text: str, width: int) -> str:
    if width <= 0:
        return ""
    acc = []
    for ch in str(text):
        if visual_width("".join(acc) + ch) > width:
            break
        acc.append(ch)
    return "".join(acc)


def fit_text(text: str, width: int) -> str:
    text = str(text)
    if visual_width(text) <= width:
        return text + " " * max(0, width - visual_width(text))
    if width <= 3:
        clipped = trim_to_width(text, width)
        return clipped + " " * max(0, width - visual_width(clipped))
    clipped = trim_to_width(text, width - 3) + "..."
    return clipped + " " * max(0, width - visual_width(clipped))


def draw_launcher_page(
    section_title: str,
    body_lines: list[str] | None = None,
    prompt_text: str | None = None,
    prompt_row: int | None = None,
) -> tuple[int, int] | None:
    clear_screen()
    width = terminal_width()
    inner = width - 2
    status_width = 27
    separator = " │ "
    left_width = inner - status_width - visual_width(separator)
    body_lines = list(body_lines or [])
    status_lines = get_status_lines()
    logo_lines = BIG_LOGO_PLAIN.splitlines()

    if prompt_text is not None:
        if prompt_row is None:
            prompt_row = len(body_lines)
        while len(body_lines) <= prompt_row:
            body_lines.append("")

    def frame_line(content: str = "") -> str:
        return "│" + fit_text(content, inner) + "│"

    def frame_line_lr(left: str = "", right: str = "", right_margin: int = 0) -> str:
        left = str(left)
        right = str(right)
        if not right:
            return frame_line(left)
        needed = visual_width(left) + visual_width(right) + right_margin + 1
        if needed <= inner:
            spaces = max(1, inner - visual_width(left) - visual_width(right) - right_margin)
            return "│" + left + (" " * spaces) + right + (" " * right_margin) + "│"
        return frame_line(left)

    def split_line(left: str = "", right: str = "") -> str:
        content = fit_text(left, left_width) + separator + fit_text(right, status_width)
        return "│" + content + "│"

    print("┌" + "─" * inner + "┐")
    last_logo_idx = len(logo_lines) - 1
    for idx, line in enumerate(logo_lines):
        if idx == last_logo_idx:
            print(frame_line_lr(" " + line, "by group 16", right_margin=20))
        else:
            print(frame_line(" " + line))
    print("├" + "─" * inner + "┤")
    print(frame_line(" " + t("tagline")))
    print("├" + "─" * inner + "┤")

    print(split_line(" " + section_title, status_lines[0]))
    rows = max(7, len(body_lines) + 1)
    for idx in range(rows):
        left = body_lines[idx] if idx < len(body_lines) else ""
        right = status_lines[idx + 1] if idx + 1 < len(status_lines) else ""
        print(split_line(" " + left if left else "", right))

    print("├" + "─" * inner + "┤")
    footer = "Type your choice and press Enter | 0 Exit / b Back | Ctrl+C Abort"
    print(frame_line(" " + footer))
    print("└" + "─" * inner + "┘")

    if prompt_text is None:
        return None

    prompt_body_row = min(prompt_row if prompt_row is not None else len(body_lines), rows - 1)
    absolute_row = 13 + prompt_body_row
    absolute_col = 4
    sys.stdout.write(f"[{absolute_row};{absolute_col}H")
    sys.stdout.write(theme_wrap(f"{C.GREEN}{prompt_text}{C.RESET}", THEME))
    sys.stdout.flush()
    return absolute_row, absolute_col + visual_width(prompt_text)

def render_logo(theme: str) -> str:
    return BIG_LOGO_PLAIN


# ---------------------------
# Language strings
# ---------------------------

LANG = "en"

I18N = {
    "en": {
        "tagline": "A lightweight deep learning project for fake news detection experiments.",
        "main_menu": "MAIN MENU",
        "single_menu": "SINGLE STAGE MODE",
        "run_all": "Run complete pipeline",
        "run_all_with_sweeps": "Run complete pipeline + full experiment sweeps",
        "run_single": "Run a single stage",
        "demo": "Demo / predict custom text",
        "exit": "Exit",
        "back": "Back",
        "choice": "Enter your choice",
        "invalid": "Invalid choice. Please try again.",
        "abort": "Aborted by user.",
        "done": "Completed successfully.",
        "pipeline_halted": "Pipeline halted because a stage failed.",
        "stage_prepare": "Prepare data",
        "stage_baseline": "Train baseline models",
        "stage_recurrent": "Train recurrent model",
        "stage_bert": "Train BERT model",
        "stage_eval": "Evaluate and generate figures",
        "stage_sweeps": "Run experiment sweeps",
        "sweeps_menu": "EXPERIMENT SWEEPS",
        "sweeps_full": "Run full recurrent sweep set",
        "sweeps_single": "Run a single sweep item",
        "sweep_base": "Base training curves",
        "sweep_loss": "Loss-function comparison",
        "sweep_lr": "Learning-rate sweep",
        "sweep_bs": "Batch-size sweep",
        "sweep_export": "Export first-N test predictions",
        "default": "default",
        "ask_param": "Enter value for",
        "confirm_defaults": "Use recommended default values",
        "continue_anyway": "Continue anyway",
        "missing_file": "Missing required file",
        "demo_missing_model": "No demo-ready model was found.",
        "demo_text": "Enter a news text",
        "demo_pred": "Prediction",
        "demo_conf": "Confidence",
        "demo_model": "Model",
        "demo_mode_text": "Interactive text input",
        "demo_mode_csv": "CSV batch labeling",
        "demo_csv_path": "Enter CSV path",
        "demo_csv_written": "Wrote labeled CSV",
        "press_enter": "Press Enter to continue...",
        "lang_select": "Select language",
        "lang_menu_note": "Choose a language to continue.",
    },
    "zh": {
        "tagline": "一个用于虚假新闻检测实验的轻量深度学习项目。",
        "main_menu": "主菜单",
        "single_menu": "单步执行模式",
        "run_all": "执行完整实验流程",
        "run_all_with_sweeps": "执行完整流程并追加完整 sweep",
        "run_single": "执行单个步骤",
        "demo": "Demo / 自定义文本预测",
        "exit": "退出",
        "back": "返回",
        "choice": "请输入选项",
        "invalid": "无效选项，请重新输入。",
        "abort": "用户已中止。",
        "done": "执行成功。",
        "pipeline_halted": "流程因某一步失败而中止。",
        "stage_prepare": "数据准备",
        "stage_baseline": "训练基线模型",
        "stage_recurrent": "训练循环模型",
        "stage_bert": "训练 BERT 模型",
        "stage_eval": "评估并生成图表",
        "stage_sweeps": "执行实验扫参与诊断",
        "sweeps_menu": "实验扫参与诊断",
        "sweeps_full": "执行完整循环模型 sweep",
        "sweeps_single": "执行单项 sweep",
        "sweep_base": "基础训练曲线",
        "sweep_loss": "损失函数对比",
        "sweep_lr": "学习率 sweep",
        "sweep_bs": "批大小 sweep",
        "sweep_export": "导出前 N 条测试预测",
        "default": "默认",
        "ask_param": "请输入参数",
        "confirm_defaults": "是否使用推荐默认值",
        "continue_anyway": "仍然继续",
        "missing_file": "缺少所需文件",
        "demo_missing_model": "未找到可用于 demo 的模型。",
        "demo_text": "请输入一段新闻文本",
        "demo_pred": "预测结果",
        "demo_conf": "置信度",
        "demo_model": "模型",
        "demo_mode_text": "直接输入文本",
        "demo_mode_csv": "CSV 批量标注",
        "demo_csv_path": "请输入 CSV 路径",
        "demo_csv_written": "已写出标注 CSV",
        "press_enter": "按回车继续...",
        "lang_select": "请选择语言",
        "lang_menu_note": "请选择继续使用的语言。",
    },
}


def t(key: str) -> str:
    return I18N[LANG][key]


ROOT = Path(__file__).resolve().parent
THEME = "plain"

DEMO_MODEL_ORDER = [
    ("logistic_regression", "Logistic Regression"),
    ("linear_svm", "Linear SVM"),
    ("lstm", "LSTM"),
    ("bert", "BERT"),
]


def clear_screen() -> None:
    # Full-screen redraw style similar to nano/vim.
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def pause() -> None:
    input(f"\n{theme_wrap(C.DIM + t('press_enter') + C.RESET, THEME)}")


def ask_value(label: str, default: str) -> str:
    text = input(
        theme_wrap(
            f"{C.YELLOW}>> {t('ask_param')} [{label}] ({t('default')}: {default}): {C.RESET}",
            THEME,
        )
    ).strip()
    return text if text else default


def ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    text = input(theme_wrap(f"{C.YELLOW}>> {question} {suffix}: {C.RESET}", THEME)).strip().lower()
    if not text:
        return default
    return text in {"y", "yes", "1", "true"}


def draw_header(section_title: str | None = None, body_lines: list[str] | None = None) -> None:
    draw_launcher_page(section_title or "Main Content", body_lines)


def launcher_input(section_title: str, body_lines: list[str], prompt_text: str) -> str:
    prompt_row = len(body_lines) + 1
    draw_launcher_page(section_title, body_lines, prompt_text=prompt_text, prompt_row=prompt_row)
    return input().strip()


def launcher_yes_no(section_title: str, body_lines: list[str], question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    text = launcher_input(section_title, body_lines, f"{question} {suffix}: ").lower()
    if not text:
        return default
    return text in {"y", "yes", "1", "true"}

def run_script(script_name: str, args: list[str], stage_title: str | None = None) -> bool:
    script_path = ROOT / script_name
    if not script_path.exists():
        print(theme_wrap(f"{C.RED}[error] {t('missing_file')}: {script_path}{C.RESET}", THEME))
        return False

    cmd = [sys.executable, str(script_path), *args]
    printable = " ".join(shlex.quote(part) for part in cmd)

    clear_screen()
    if stage_title:
        print(f"[stage] {stage_title}")
    print(f"[run] {printable}")
    print("=" * 72)
    print()

    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        return True
    except subprocess.CalledProcessError as exc:
        print(f"\n[error] Script failed with exit code {exc.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[aborted] {t('abort')}")
        return False


def warn_missing(path: Path) -> bool:
    if path.exists():
        return True
    print(theme_wrap(f"{C.RED}[missing] {t('missing_file')}: {path}{C.RESET}", THEME))
    return ask_yes_no(t("continue_anyway"), default=False)


def stage_prepare(use_defaults: bool = False) -> bool:
    input_path = "data/raw/fake_news.csv" if use_defaults else ask_value("input", "data/raw/fake_news.csv")
    label_map = "0:real,1:fake" if use_defaults else ask_value("label-map", "0:real,1:fake")
    return run_script("prepare_data.py", ["--input", input_path, "--label-map", label_map], stage_title=t("stage_prepare"))


def stage_baseline(use_defaults: bool = False) -> bool:
    if not warn_missing(ROOT / "data" / "splits" / "split_table.csv"):
        return False

    text_column = "clean_text" if use_defaults else ask_value("text-column", "clean_text")
    max_features = "5000" if use_defaults else ask_value("max-features", "5000")
    positive_label = "fake" if use_defaults else ask_value("positive-label", "fake")
    return run_script(
        "train_baselines.py",
        ["--text-column", text_column, "--max-features", max_features, "--positive-label", positive_label],
        stage_title=t("stage_baseline"),
    )


def stage_recurrent(use_defaults: bool = False) -> bool:
    if not warn_missing(ROOT / "data" / "splits" / "split_table.csv"):
        return False

    text_column = "clean_text" if use_defaults else ask_value("text-column", "clean_text")
    model = "lstm" if use_defaults else ask_value("model", "lstm")
    loss_type = "cross_entropy" if use_defaults else ask_value("loss-type", "cross_entropy")
    epochs = "5" if use_defaults else ask_value("epochs", "5")
    positive_label = "fake" if use_defaults else ask_value("positive-label", "fake")
    return run_script(
        "train_recurrent.py",
        [
            "--text-column", text_column,
            "--model", model,
            "--loss-type", loss_type,
            "--epochs", epochs,
            "--positive-label", positive_label,
        ],
        stage_title=t("stage_recurrent"),
    )


def stage_bert(use_defaults: bool = False) -> bool:
    if not warn_missing(ROOT / "data" / "splits" / "split_table.csv"):
        return False

    text_column = "clean_text" if use_defaults else ask_value("text-column", "clean_text")
    model_name = "bert-base-uncased" if use_defaults else ask_value("model-name", "bert-base-uncased")
    epochs = "3" if use_defaults else ask_value("epochs", "3")
    positive_label = "fake" if use_defaults else ask_value("positive-label", "fake")
    return run_script(
        "train_bert.py",
        [
            "--text-column", text_column,
            "--model-name", model_name,
            "--epochs", epochs,
            "--positive-label", positive_label,
        ],
        stage_title=t("stage_bert"),
    )


def stage_eval(use_defaults: bool = False) -> bool:
    text_column = "clean_text" if use_defaults else ask_value("text-column", "clean_text")
    positive_label = "fake" if use_defaults else ask_value("positive-label", "fake")
    return run_script(
        "evaluate_results.py",
        ["--text-column", text_column, "--positive-label", positive_label],
        stage_title=t("stage_eval"),
    )


def stage_experiment_sweeps(mode: str = "full", use_defaults: bool = False) -> bool:
    if not warn_missing(ROOT / "data" / "splits" / "split_table.csv"):
        return False

    text_column = "clean_text" if use_defaults else ask_value("text-column", "clean_text")
    model = "lstm" if use_defaults else ask_value("model", "lstm")
    positive_label = "fake" if use_defaults else ask_value("positive-label", "fake")
    epochs = "5" if use_defaults else ask_value("epochs", "5")
    base_loss = "cross_entropy" if use_defaults else ask_value("base-loss", "cross_entropy")
    alt_loss = "weighted_ce" if use_defaults else ask_value("alt-loss", "weighted_ce")
    base_lr = "0.001" if use_defaults else ask_value("base-learning-rate", "0.001")
    base_bs = "32" if use_defaults else ask_value("base-batch-size", "32")
    learning_rates = "0.01,0.001,0.0001" if use_defaults else ask_value("learning-rates", "0.01,0.001,0.0001")
    batch_sizes = "16,32,64" if use_defaults else ask_value("batch-sizes", "16,32,64")
    first_n = "100" if use_defaults else ask_value("first-n", "100")

    args = [
        "--mode", mode,
        "--text-column", text_column,
        "--model", model,
        "--positive-label", positive_label,
        "--epochs", epochs,
        "--base-loss", base_loss,
        "--alt-loss", alt_loss,
        "--base-learning-rate", base_lr,
        "--base-batch-size", base_bs,
        "--learning-rates", learning_rates,
        "--batch-sizes", batch_sizes,
        "--first-n", first_n,
    ]
    return run_script("run_experiment_sweeps.py", args, stage_title=t("stage_sweeps"))


def sweeps_menu() -> None:
    while True:
        body_lines = [
            "[1] " + t("sweeps_full"),
            "[2] " + t("sweep_base"),
            "[3] " + t("sweep_loss"),
            "[4] " + t("sweep_lr"),
            "[5] " + t("sweep_bs"),
            "[6] " + t("sweep_export"),
            "",
            "[b] " + t("back"),
        ]
        choice = launcher_input(t("sweeps_menu"), body_lines, f"{t('choice')} [1-6/b]: ").lower()
        if choice == "b":
            return
        mode_map = {
            "1": "full",
            "2": "base",
            "3": "loss",
            "4": "learning_rate",
            "5": "batch_size",
            "6": "export_first_n",
        }
        if choice in mode_map:
            use_defaults = ask_yes_no(t("confirm_defaults"), default=True)
            ok = stage_experiment_sweeps(mode=mode_map[choice], use_defaults=use_defaults)
            if ok:
                print(theme_wrap(f"\n{C.GREEN}[done] {t('done')}{C.RESET}", THEME))
            pause()
        else:
            print(theme_wrap(f"{C.RED}{t('invalid')}{C.RESET}", THEME))
            pause()


def clean_demo_text(text: str) -> str:
    try:
        mod = importlib.import_module("utils.text_utils")
        if hasattr(mod, "basic_clean_text"):
            return str(mod.basic_clean_text(text))
    except Exception:
        pass
    return str(text)


def resolve_demo_model_paths() -> dict[str, Path | None]:
    def _first_existing(candidates: list[Path]) -> Path | None:
        for path in candidates:
            if path.exists():
                return path
        return None

    bert_dir = None
    bert_root = ROOT / "outputs" / "models" / "bert"
    if bert_root.exists():
        preferred = bert_root / "bert_base_uncased"
        if preferred.exists() and preferred.is_dir():
            bert_dir = preferred
        else:
            for child in sorted(bert_root.iterdir()):
                if child.is_dir() and (child / "config.json").exists():
                    bert_dir = child
                    break

    return {
        "logistic_regression": _first_existing([
            ROOT / "outputs" / "models" / "baselines" / "logistic_regression_clean_text.joblib",
            ROOT / "outputs" / "models" / "sparse" / "logistic_regression_clean_text.joblib",
        ]),
        "linear_svm": _first_existing([
            ROOT / "outputs" / "models" / "baselines" / "linear_svm_clean_text.joblib",
            ROOT / "outputs" / "models" / "sparse" / "linear_svm_clean_text.joblib",
        ]),
        "lstm": _first_existing([
            ROOT / "outputs" / "models" / "recurrent" / "lstm_ce.pt",
        ]),
        "bert": bert_dir,
    }


def resolve_demo_label_names() -> list[str]:
    split_path = ROOT / "data" / "splits" / "split_table.csv"
    if pd is not None and split_path.exists():
        try:
            labels = sorted(pd.read_csv(split_path)["label_name"].astype(str).unique().tolist())
            if labels:
                return labels
        except Exception:
            pass
    return ["fake", "real"]


def parse_recurrent_alias(name: str) -> tuple[str, bool]:
    key = str(name).lower().strip()
    mapping = {
        "rnn": ("rnn", False),
        "gru": ("gru", False),
        "lstm": ("lstm", False),
        "bigru": ("gru", True),
        "bilstm": ("lstm", True),
    }
    return mapping.get(key, ("lstm", False))


def resolve_demo_assets() -> dict[str, Any]:
    paths = resolve_demo_model_paths()
    label_names = resolve_demo_label_names()
    assets: dict[str, Any] = {"label_names": label_names}
    if joblib is not None:
        if paths.get("logistic_regression") is not None:
            try:
                assets["logistic_regression"] = joblib.load(paths["logistic_regression"])
            except Exception:
                pass
        if paths.get("linear_svm") is not None:
            try:
                assets["linear_svm"] = joblib.load(paths["linear_svm"])
            except Exception:
                pass

    try:
        import torch
        from utils.train_utils import RecurrentTextClassifier, Vocab, load_sequence_classifier

        if paths.get("lstm") is not None:
            ckpt = torch.load(paths["lstm"], map_location="cpu")
            stoi = ckpt.get("vocab", {}) or {}
            max_idx = max(stoi.values()) if stoi else 1
            itos = [""] * (max_idx + 1)
            for token, idx in stoi.items():
                if 0 <= int(idx) < len(itos):
                    itos[int(idx)] = token
            vocab = Vocab(stoi=stoi, itos=itos)
            cfg = ckpt.get("config", {}) or {}
            rnn_type, bidirectional = parse_recurrent_alias(cfg.get("model", "lstm"))
            model = RecurrentTextClassifier(
                vocab_size=len(vocab.itos),
                embedding_dim=int(cfg.get("embedding_dim", 128)),
                hidden_dim=int(cfg.get("hidden_dim", 128)),
                num_classes=len(label_names),
                rnn_type=rnn_type,
                num_layers=int(cfg.get("num_layers", 1)),
                dropout=float(cfg.get("dropout", 0.2)),
                bidirectional=bidirectional,
                pad_id=vocab.pad_id,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            assets["lstm"] = {
                "model": model,
                "vocab": vocab,
                "max_length": int(cfg.get("max_length", 256)),
            }

        if paths.get("bert") is not None:
            tokenizer, model = load_sequence_classifier(
                str(paths["bert"]),
                num_labels=len(label_names),
                local_files_only=True,
            )
            model.eval()
            assets["bert"] = {
                "tokenizer": tokenizer,
                "model": model,
                "max_length": 256,
            }
    except Exception:
        pass
    return assets


def _device_for_demo() -> str:
    return current_device_label()


def _predict_sparse_batch(model, texts: list[str]) -> tuple[list[str], list[float | None]]:
    cleaned = [clean_demo_text(text) for text in texts]
    preds = [str(x) for x in model.predict(cleaned)]
    scores: list[float | None] = [None] * len(preds)
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(cleaned)
            scores = [float(max(row)) for row in probs]
        except Exception:
            pass
    return preds, scores


def _predict_lstm_batch(bundle: dict[str, Any], texts: list[str], label_names: list[str]) -> tuple[list[str], list[float | None]]:
    import torch
    from utils.text_utils import basic_tokenize

    model = bundle["model"]
    vocab = bundle["vocab"]
    max_length = int(bundle.get("max_length", 256))
    device_label = _device_for_demo()
    device = torch.device(device_label if device_label != "mps" or getattr(torch.backends, "mps", None) else "cpu")
    model = model.to(device)

    preds: list[str] = []
    scores: list[float | None] = []
    with torch.no_grad():
        for text in texts:
            cleaned = clean_demo_text(text)
            tokens = basic_tokenize(cleaned)[:max_length]
            input_ids = vocab.encode(tokens) if tokens else [vocab.pad_id]
            seq = torch.tensor([input_ids], dtype=torch.long, device=device)
            lengths = torch.tensor([len(input_ids)], dtype=torch.long, device=device)
            logits = model(seq, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            preds.append(label_names[pred_idx])
            scores.append(float(probs.max()))
    return preds, scores


def _predict_bert_batch(bundle: dict[str, Any], texts: list[str], label_names: list[str]) -> tuple[list[str], list[float | None]]:
    import torch

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    max_length = int(bundle.get("max_length", 256))
    device_label = _device_for_demo()
    device = torch.device(device_label if device_label != "mps" or getattr(torch.backends, "mps", None) else "cpu")
    model = model.to(device)

    cleaned = [clean_demo_text(text) for text in texts]
    preds: list[str] = []
    scores: list[float | None] = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start:start + batch_size]
            enc = tokenizer(chunk, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_ids = probs.argmax(axis=1)
            preds.extend([label_names[int(i)] for i in pred_ids])
            scores.extend([float(row.max()) for row in probs])
    return preds, scores


def predict_all_demo_models(texts: list[str]) -> list[dict[str, Any]]:
    assets = resolve_demo_assets()
    label_names = assets.get("label_names", ["fake", "real"])
    n = len(texts)
    rows = [dict() for _ in range(n)]

    model_batches: dict[str, tuple[list[str], list[float | None]]] = {}
    if assets.get("logistic_regression") is not None:
        model_batches["logistic_regression"] = _predict_sparse_batch(assets["logistic_regression"], texts)
    if assets.get("linear_svm") is not None:
        model_batches["linear_svm"] = _predict_sparse_batch(assets["linear_svm"], texts)
    if assets.get("lstm") is not None:
        model_batches["lstm"] = _predict_lstm_batch(assets["lstm"], texts, label_names)
    if assets.get("bert") is not None:
        model_batches["bert"] = _predict_bert_batch(assets["bert"], texts, label_names)

    for idx in range(n):
        vote_payload: dict[str, str] = {}
        for model_key, _display in DEMO_MODEL_ORDER:
            labels, scores = model_batches.get(model_key, (["N/A"] * n, [None] * n))
            label = labels[idx] if idx < len(labels) else "N/A"
            score = scores[idx] if idx < len(scores) else None
            rows[idx][model_key] = label
            rows[idx][f"{model_key}_score"] = score
            if label != "N/A":
                vote_payload[model_key] = label
        rows[idx]["consensus_label"] = compute_consensus_label(vote_payload)
    return rows


def compute_consensus_label(labels_by_model: dict[str, str]) -> str:
    valid = [label for label in labels_by_model.values() if label and label != "N/A"]
    if not valid:
        return "N/A"
    counts = Counter(valid)
    max_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == max_count}
    if len(tied) == 1:
        return next(iter(tied))
    for key in ["bert", "lstm", "logistic_regression", "linear_svm"]:
        label = labels_by_model.get(key)
        if label in tied:
            return label
    return sorted(tied)[0]


def run_demo_interactive() -> None:
    clear_screen()
    print(f"[demo] {t('demo_mode_text')}")
    text = input(theme_wrap(f"{C.YELLOW}>> {t('demo_text')}: {C.RESET}", THEME)).strip()
    if not text:
        print(theme_wrap(f"{C.YELLOW}{t('abort')}{C.RESET}", THEME))
        pause()
        return

    row = predict_all_demo_models([text])[0]
    print()
    for model_key, display_name in DEMO_MODEL_ORDER:
        print(theme_wrap(f"{C.GREEN}{display_name}: {row.get(model_key, 'N/A')}{C.RESET}", THEME))
    print(theme_wrap(f"{C.GREEN}Majority Vote: {row.get('consensus_label', 'N/A')}{C.RESET}", THEME))
    pause()


def run_demo_csv() -> None:
    clear_screen()
    print(f"[demo] {t('demo_mode_csv')}")
    if pd is None:
        print(theme_wrap(f"{C.RED}pandas is not available in this environment.{C.RESET}", THEME))
        pause()
        return
    csv_path_text = input(theme_wrap(f"{C.YELLOW}>> {t('demo_csv_path')}: {C.RESET}", THEME)).strip().strip('"')
    if not csv_path_text:
        print(theme_wrap(f"{C.YELLOW}{t('abort')}{C.RESET}", THEME))
        pause()
        return
    csv_path = Path(csv_path_text)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        print(theme_wrap(f"{C.RED}[missing] {csv_path}{C.RESET}", THEME))
        pause()
        return
    df = pd.read_csv(csv_path)
    if df.empty or len(df.columns) < 1:
        print(theme_wrap(f"{C.RED}[error] CSV must contain at least one text column.{C.RESET}", THEME))
        pause()
        return
    first_col = df.columns[0]
    texts = df[first_col].fillna("").astype(str).tolist()
    predictions = predict_all_demo_models(texts)
    analysis_df = pd.DataFrame({
        "logistic_regression": [row.get("logistic_regression") for row in predictions],
        "linear_svm": [row.get("linear_svm") for row in predictions],
        "lstm": [row.get("lstm") for row in predictions],
        "bert": [row.get("bert") for row in predictions],
        "logistic_regression_score": [row.get("logistic_regression_score") for row in predictions],
        "linear_svm_score": [row.get("linear_svm_score") for row in predictions],
        "lstm_score": [row.get("lstm_score") for row in predictions],
        "bert_score": [row.get("bert_score") for row in predictions],
        "consensus_label": [row.get("consensus_label") for row in predictions],
    })
    primary = df[[first_col]]
    rest = df.drop(columns=[first_col])
    result = pd.concat([primary, analysis_df, rest], axis=1)
    out_path = csv_path.with_name(f"{csv_path.stem}_labeled{csv_path.suffix or '.csv'}")
    result.to_csv(out_path, index=False)
    print(theme_wrap(f"{C.GREEN}[ok] {t('demo_csv_written')}: {out_path}{C.RESET}", THEME))
    pause()


def run_demo() -> None:
    draw_header(t("demo"), ["[1] " + t("demo_mode_text"), "[2] " + t("demo_mode_csv"), "", "[0] " + t("back")])
    if joblib is None:
        print(theme_wrap(f"{C.RED}joblib is not available in this environment.{C.RESET}", THEME))
        pause()
        return

    paths = resolve_demo_model_paths()
    if not any(paths.values()):
        print(theme_wrap(f"{C.RED}{t('demo_missing_model')}{C.RESET}", THEME))
        print(theme_wrap(f"{C.DIM}Train at least one model first, then try again.{C.RESET}", THEME))
        pause()
        return

    body_lines = [
        "[1] " + t("demo_mode_text"),
        "[2] " + t("demo_mode_csv"),
        "",
        "[0] " + t("back"),
    ]
    choice = launcher_input(t("demo"), body_lines, f"{t('choice')} [1-2/0]: ").lower()
    if choice in {"0", "b"}:
        return
    if choice == "1":
        run_demo_interactive()
        return
    if choice == "2":
        run_demo_csv()
        return

    print(theme_wrap(f"{C.RED}{t('invalid')}{C.RESET}", THEME))
    pause()


STAGES: dict[str, Callable[[bool], bool]] = {
    "1": stage_prepare,
    "2": stage_baseline,
    "3": stage_recurrent,
    "4": stage_bert,
    "5": stage_eval,
}



def run_pipeline_sequence(use_defaults: bool, include_full_sweeps: bool = False) -> bool:
    for step in ["1", "2", "3", "4", "5"]:
        ok = STAGES[step](use_defaults=use_defaults)
        if not ok:
            return False
    if include_full_sweeps:
        ok = stage_experiment_sweeps(mode="full", use_defaults=use_defaults)
        if not ok:
            return False
    return True


def run_all_pipeline(include_full_sweeps: bool = False) -> None:
    header_key = "run_all_with_sweeps" if include_full_sweeps else "run_all"
    body_lines = ["Ready to launch the selected workflow.", "Execution logs will switch to raw terminal mode."]
    use_defaults = launcher_yes_no(t(header_key), body_lines, t("confirm_defaults"), default=True)
    ok = run_pipeline_sequence(use_defaults=use_defaults, include_full_sweeps=include_full_sweeps)
    if not ok:
        print(theme_wrap(f"\n{C.RED}[halted] {t('pipeline_halted')}{C.RESET}", THEME))
        pause()
        return
    print(theme_wrap(f"\n{C.GREEN}[done] {t('done')}{C.RESET}", THEME))
    pause()


def single_stage_menu() -> None:
    while True:
        body_lines = [
            "[1] " + t("stage_prepare"),
            "[2] " + t("stage_baseline"),
            "[3] " + t("stage_recurrent"),
            "[4] " + t("stage_bert"),
            "[5] " + t("stage_eval"),
            "[6] " + t("stage_sweeps"),
            "",
            "[b] " + t("back"),
        ]
        choice = launcher_input(t("single_menu"), body_lines, f"{t('choice')} [1-6/b]: ").lower()
        if choice == "b":
            return
        if choice in STAGES:
            ok = STAGES[choice](use_defaults=False)
            if ok:
                print(theme_wrap(f"\n{C.GREEN}[done] {t('done')}{C.RESET}", THEME))
            pause()
        elif choice == "6":
            sweeps_menu()
        else:
            print(theme_wrap(f"{C.RED}{t('invalid')}{C.RESET}", THEME))
            pause()


def main_menu() -> None:
    while True:
        body_lines = [
            "[1] " + t("run_all"),
            "[2] " + t("run_all_with_sweeps"),
            "[3] " + t("run_single"),
            "[4] " + t("demo"),
            "",
            "[0] " + t("exit"),
        ]
        choice = launcher_input("Main Content", body_lines, f"{t('choice')} [0-4]: ").lower()
        if choice == "0":
            clear_screen()
            print("Goodbye! / 再见！")
            return
        if choice == "1":
            run_all_pipeline(include_full_sweeps=False)
        elif choice == "2":
            run_all_pipeline(include_full_sweeps=True)
        elif choice == "3":
            single_stage_menu()
        elif choice == "4":
            run_demo()
        else:
            print(theme_wrap(f"{C.RED}{t('invalid')}{C.RESET}", THEME))
            pause()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive orchestrator for News Authenticator.")
    parser.add_argument("--lang", choices=["en", "zh"], help="Start directly in a language.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument("--theme", choices=["plain", "aubergine"], default="plain", help="Visual theme.")
    return parser.parse_args()


def language_screen() -> list[str]:
    return [
        I18N["en"]["lang_menu_note"],
        "",
        "[1] English (Default)",
        "[2] 中文",
        "",
        "[0] Exit",
    ]


def main() -> None:
    global LANG, THEME
    args = parse_args()

    if args.no_color:
        for attr in ["RESET", "BOLD", "WHITE", "CYAN", "MAGENTA", "GREEN", "YELLOW", "RED", "BLUE", "DIM"]:
            setattr(C, attr, "")

    THEME = args.theme

    try:
        if args.lang:
            LANG = args.lang
        else:
            body_lines = language_screen()
            lang_choice = launcher_input("Language Selection", body_lines, f"{I18N['en']['lang_select']} [1/2/0]: ").strip().lower()
            if lang_choice in {"0", "q", "quit", "exit"}:
                clear_screen()
                print("Goodbye! / 再见！")
                return
            LANG = "zh" if lang_choice == "2" else "en"

        main_menu()
    except KeyboardInterrupt:
        clear_screen()
        print(theme_wrap(f"{C.YELLOW}[aborted] {t('abort')}{C.RESET}", THEME))


if __name__ == "__main__":
    main()
