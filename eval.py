from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agent import MultiToolAgent

console = Console()


@dataclass
class EvalCase:
    name: str
    prompt: str
    expected_tool: str


@dataclass
class EvalResult:
    name: str
    expected_tool: str
    used_tools: list[str]
    passed: bool
    answer: str


CASES = [
    EvalCase(
        name="summarize",
        prompt=(
            "Use the summarize_text tool with max_sentences=2 on this text: "
            "'Distributed systems are hard because failures are partial. "
            "A node can be slow, unreachable, or return stale data. "
            "Good systems use timeouts, retries, and idempotency. "
            "Observability matters because logs alone rarely explain failures.'"
        ),
        expected_tool="summarize_text",
    ),
    EvalCase(
        name="keywords",
        prompt=(
            "Use the extract_keywords tool with limit=5 on this text: "
            "'Kubernetes helps teams deploy containers. "
            "Container orchestration improves reliability and scaling.'"
        ),
        expected_tool="extract_keywords",
    ),
    EvalCase(
        name="ideas",
        prompt=(
            "Use the generate_ideas tool with count=3 for the topic "
            "'sustainable coffee shops'."
        ),
        expected_tool="generate_ideas",
    ),
    EvalCase(
        name="word-count",
        prompt=(
            "Use the word_count tool on this text: "
            "'Small steps compound over time. Consistency matters.'"
        ),
        expected_tool="word_count",
    ),
]


def run_eval(agent: MultiToolAgent | None = None) -> list[EvalResult]:
    load_dotenv(".env")
    agent = agent or MultiToolAgent()
    results: list[EvalResult] = []

    for case in CASES:
        answer = agent.run(case.prompt)
        used_tools = [item.name for item in agent.trace]
        has_error = any(item.error for item in agent.trace)
        passed = case.expected_tool in used_tools and not has_error
        results.append(
            EvalResult(
                name=case.name,
                expected_tool=case.expected_tool,
                used_tools=used_tools,
                passed=passed,
                answer=answer,
            )
        )

    return results


def main() -> int:
    results = run_eval()
    table = Table(title="agent eval")
    table.add_column("case")
    table.add_column("expected")
    table.add_column("used")
    table.add_column("passed")

    for result in results:
        table.add_row(
            result.name,
            result.expected_tool,
            ", ".join(result.used_tools) or "-",
            "yes" if result.passed else "no",
        )

    console.print(table)
    passed = sum(result.passed for result in results)
    console.print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
