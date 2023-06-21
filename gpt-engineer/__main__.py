import json
import logging
import os
import shutil
import asyncio

from pathlib import Path

import typer

from gpt_engineer.ai import AI
from gpt_engineer.db import DB, DBs
from gpt_engineer.steps import STEPS

app = typer.Typer()

async def async_main(
    project_path: str,
    delete_existing: bool,
    model: str,
    temperature: float,
    steps_config: str,
    verbose: bool,
    run_prefix: str,
):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    input_path = Path(project_path).absolute()
    memory_path = input_path / f"{run_prefix}memory"
    workspace_path = input_path / f"{run_prefix}workspace"

    if delete_existing:
        # Delete files and subdirectories in paths
        shutil.rmtree(memory_path, ignore_errors=True)
        shutil.rmtree(workspace_path, ignore_errors=True)

    ai = AI(
        model=model,
        temperature=temperature,
    )

    dbs = DBs(
        memory=DB(memory_path),
        logs=DB(memory_path / "logs"),
        input=DB(input_path),
        workspace=DB(workspace_path),
        identity=DB(Path(os.path.curdir) / "identity"),
    )

    for step in STEPS[steps_config]:
        messages = await step(ai, dbs)
        dbs.logs[step.__name__] = json.dumps(messages)


@app.command()
def main(
    project_path: str = typer.Argument("example", help="path"),
    delete_existing: bool = typer.Argument(False, help="delete existing files"),
    model: str = "gpt-4",
    temperature: float = 0.1,
    steps_config: str = "default",
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    run_prefix: str = typer.Option(
        "",
        help=(
            "run prefix, if you want to run multiple variants of the same project and "
            "later compare them"
        ),
    ),
):
    asyncio.run(async_main(project_path, delete_existing, model, temperature, steps_config, verbose, run_prefix))


if __name__ == "__main__":
    app()
