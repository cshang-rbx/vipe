from __future__ import annotations

import argparse
import subprocess
import sys

from datetime import datetime, timezone
from os import environ
from pathlib import Path

import gradio as gr
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "gradio_runs"
EXAMPLE_DIR = REPO_ROOT / "assets" / "examples"
TORCH_LIB_DIR = Path(torch.__file__).resolve().parent / "lib"


def list_pipeline_names() -> list[str]:
    pipeline_dir = REPO_ROOT / "configs" / "pipeline"
    return sorted(path.stem for path in pipeline_dir.glob("*.yaml"))


def list_example_videos() -> list[list[str]]:
    if not EXAMPLE_DIR.exists():
        return []
    return [[str(path)] for path in sorted(EXAMPLE_DIR.glob("*.mp4"))]


def choose_preview_video(output_dir: Path) -> str | None:
    preview_patterns = [
        output_dir / "vipe" / "*_vis.mp4",
        output_dir / "rgb" / "*.mp4",
    ]
    for pattern in preview_patterns:
        matches = sorted(pattern.parent.glob(pattern.name))
        if matches:
            return str(matches[0])
    return None


def summarize_output(output_dir: Path, return_code: int, command: list[str]) -> str:
    preview_path = choose_preview_video(output_dir)
    lines = [
        f"Return code: `{return_code}`",
        f"Output directory: `{output_dir}`",
        f"Command: `{' '.join(command)}`",
    ]
    if preview_path is not None:
        lines.append(f"Preview video: `{preview_path}`")
    else:
        lines.append("Preview video: not found. Turn on visualization to produce `*_vis.mp4` output.")
    return "\n".join(lines)


def run_vipe_demo(
    video_path: str | None,
    pipeline_name: str,
    visualize: bool,
    output_root: str,
) -> tuple[str, str, str | None]:
    if not video_path:
        raise gr.Error("Upload a video or choose one of the bundled examples first.")

    input_path = Path(video_path)
    if not input_path.exists():
        raise gr.Error(f"Input video does not exist: {input_path}")

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{input_path.stem}_{pipeline_name}_{run_stamp}"
    run_output_dir = Path(output_root).expanduser().resolve() / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "vipe.cli.main",
        "infer",
        str(input_path.resolve()),
        "--output",
        str(run_output_dir),
        "--pipeline",
        pipeline_name,
    ]
    if visualize:
        command.append("--visualize")

    ld_library_path = str(TORCH_LIB_DIR)
    if environ.get("LD_LIBRARY_PATH"):
        ld_library_path = f"{ld_library_path}:{environ['LD_LIBRARY_PATH']}"

    env = dict(
        environ,
        PYTHONUNBUFFERED="1",
        HYDRA_FULL_ERROR="1",
        LD_LIBRARY_PATH=ld_library_path,
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    logs = completed.stdout
    if completed.stderr:
        if logs:
            logs += "\n"
        logs += completed.stderr

    summary = summarize_output(run_output_dir, completed.returncode, command)
    preview_path = choose_preview_video(run_output_dir)

    if completed.returncode != 0:
        summary = f"## ViPE failed\n\n{summary}"
        return summary, logs, preview_path

    return summary, logs, preview_path


def build_demo(default_output_root: Path) -> gr.Blocks:
    pipeline_names = list_pipeline_names()
    if not pipeline_names:
        raise RuntimeError(f"No pipeline configs were found under {REPO_ROOT / 'configs' / 'pipeline'}")
    default_pipeline = "default" if "default" in pipeline_names else pipeline_names[0]

    with gr.Blocks(title="ViPE Gradio Demo") as demo:
        gr.Markdown(
            """
            # ViPE Demo
            Upload a video and run the existing `vipe infer` pipeline through a lightweight Gradio UI.

            This demo expects the repository to be installed in the current Python environment, with CUDA and the
            compiled ViPE extensions available.
            """
        )

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(
                    label="Input video",
                    sources=["upload"],
                )
                pipeline_name = gr.Dropdown(
                    choices=pipeline_names,
                    value=default_pipeline,
                    label="Pipeline",
                )
                visualize = gr.Checkbox(
                    value=True,
                    label="Save visualization video",
                    info="Recommended. This produces the `vipe/*_vis.mp4` preview used by the demo output.",
                )
                output_root = gr.Textbox(
                    value=str(default_output_root),
                    label="Output root directory",
                )
                run_button = gr.Button("Run ViPE", variant="primary")

            with gr.Column():
                summary = gr.Markdown(label="Run summary")
                preview_video = gr.Video(label="Preview output")
                logs = gr.Textbox(label="Logs", lines=18, max_lines=30)

        examples = list_example_videos()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[input_video],
                label="Bundled example videos",
            )

        run_button.click(
            fn=run_vipe_demo,
            inputs=[input_video, pipeline_name, visualize, output_root],
            outputs=[summary, logs, preview_video],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for ViPE.")
    parser.add_argument("--server-name", default="0.0.0.0", help="Host interface to bind the Gradio app to.")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for the Gradio app.")
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Default directory where ViPE demo outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_root = Path(args.output_root).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    demo = build_demo(target_root)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[str(REPO_ROOT), str(target_root)],
    )


if __name__ == "__main__":
    main()
