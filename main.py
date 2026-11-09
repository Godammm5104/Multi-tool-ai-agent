from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agent import DEFAULT_MODEL, MultiToolAgent

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-tool CLI agent.")
    parser.add_argument(
        "--model",
        default=os.getenv("IO_MODEL", DEFAULT_MODEL),
        help="Model ID to use.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Maximum number of agent/tool steps.",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print every tool call after each answer.",
    )
    return parser


def print_trace(agent: MultiToolAgent) -> None:
    if not agent.trace:
        console.print("[dim]no tools used.[/]")
        return

    table = Table(title="tool trace")
    table.add_column("step", justify="right")
    table.add_column("tool")
    table.add_column("arguments")
    table.add_column("result")
    table.add_column("error")

    for item in agent.trace:
        table.add_row(
            str(item.step),
            item.name,
            str(item.arguments)[:80],
            item.result.replace("\n", " ")[:100],
            item.error or "",
        )
    console.print(table)


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()

    if not os.getenv("IO_API_KEY"):
        console.print(
            "[red]IO_API_KEY is not set.[/] Copy .env.example to .env and add your key."
        )
        return 1

    agent = MultiToolAgent(model=args.model, max_steps=args.max_steps)

    console.print("[bold]Multi-tool agent[/]")
    console.print(f"model: [cyan]{agent.model}[/]")
    console.print("commands: /trace, /clear, /help, /exit")
    console.print()

    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            return 0

        if not user_message:
            continue

        command = user_message.lower()
        if command in {"/exit", "/quit"}:
            console.print("bye.")
            return 0
        if command == "/clear":
            agent.reset()
            console.print("[dim]trace cleared.[/]")
            continue
        if command == "/trace":
            print_trace(agent)
            continue
        if command == "/help":
            console.print("commands: /trace, /clear, /help, /exit")
            continue

        try:
            answer = agent.run(user_message)
            console.print(f"[bold green]assistant>[/] {answer}")
            if args.show_trace:
                print_trace(agent)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]error:[/] {exc}")


if __name__ == "__main__":
    sys.exit(main())
