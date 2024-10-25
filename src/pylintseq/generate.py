import functools
import jsonlines
import sys
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from argparse import ArgumentParser
import multiprocessing
import gc
import psutil
import warnings

warnings.filterwarnings("ignore")

from .utils import *

"""Multithreaded synthetic edit sequence generation in Python with LintSeq."""

# Set 'spawn' method to prevent memory leaks when forking large objects
multiprocessing.set_start_method("spawn", force=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Random seed used during synthetic data generation.",
    )
    parser.add_argument(
        "-p",
        "--name_or_path",
        default=None,
        type=str,
        help="Path to source JSONLines file.",
    )
    parser.add_argument(
        "-d",
        "--dest_dir",
        default=None,
        type=str,
        help="""Destination directory for synthetically generated data.""",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        default="all",
        help="""Number of samples to process. If passed as an integer, subsamples the dataset 
        without replacement. Otherwise, processes all data in the source file.""",
    )
    parser.add_argument(
        "-c",
        "--num_workers",
        default=8,
        type=int,
        help="Number of parallel workers to use during synthetic data generation.",
    )
    parser.add_argument(
        "--code_data_field",
        default="response",
        help="Name of example code data field in the source dataset.",
    )
    parser.add_argument(
        "--prompt_data_field",
        default="instruction",
        help="Name of prompt data field in the source dataset (if any). Pass as `None` if not defined.",
    )
    parser.add_argument(
        "-s",
        "--num_edit_paths_per_sample",
        default=1,
        type=int,
        help="How many edit paths should we (independently) sample per example in the dataset?",
    )

    args = parser.parse_args()

    if args.name_or_path is None:
        raise ValueError(f"Please specify a source file of code data to lint")
    elif not os.path.exists(args.name_or_path):
        raise ValueError(
            f"Cannot find {args.name_or_path} in your file system. Are you sure this file exists?"
        )

    assert args.name_or_path[args.name_or_path.find(".") + 1 :] in [
        "jsonl",
        "jsonlines",
    ], "Source data must be in jsonlines format!"

    if args.prompt_data_field == "None":
        args.prompt_data_field = None

    if not args.dest_dir:
        args.dest_dir = os.getcwd()
    else:
        os.makedirs(args.dest_dir, exist_ok=True)

    source_file = args.name_or_path.split("/")[-1]
    args.dest = os.path.join(
        args.dest_dir,
        source_file[: source_file.find(".")]
        + f"_{args.num_samples}_{args.num_edit_paths_per_sample}_{args.seed}_vec.jsonl",
    )
    if os.path.exists(args.dest):
        raise ValueError(
            f"A file already exists at the destination {args.dest}. Please remove this file to continue!"
        )
    print(f"Dumping pylintseq outputs to: {args.dest}")
    return args


def subprocess_task(start_i, total_samples, args, shared_df, samples):
    """
    Batch generation of edit paths for each process to minimize memory usage.
    Uses shared memory to avoid duplicating large data objects.
    """
    data = []
    df_slice = shared_df.iloc[samples[start_i : start_i + total_samples]].copy()

    for i in range(total_samples):
        index = samples[start_i + i]
        code_as_text = df_slice[args.code_data_field].iloc[i]
        code_as_text = strip_chain_of_thought(code_as_text)

        for _ in range(args.num_edit_paths_per_sample):
            edit_path = lintseq_backward_sampling_pythonic(
                code_as_text,
                children_per_round=1,
                top_k=1,
                max_population_size=1,
                max_depth=512,
                indent_bias_sampling_factor=1,
                ignore_imports=False,
                ignore_comments=True,
                ignore_global_defs=True,
                ignore_init_errors=False,
            )

            if edit_path is None:
                continue

            edit_sequence = edit_path[0][0]
            _, diff_seq = inflate_edit_path(code_as_text, edit_sequence)

            datum = {
                "edit_path": diff_seq,
                "index": int(index),
                "source_file": args.name_or_path,
                f"source_{args.code_data_field}": df_slice[args.code_data_field].iloc[
                    i
                ],
            }

            if not args.prompt_data_field is None:
                datum[f"source_{args.prompt_data_field}"] = df_slice[
                    args.prompt_data_field
                ].iloc[i]
            data.append(datum)

        # Clear large variables to release memory immediately
        del code_as_text, edit_path
    del df_slice
    gc.collect()  # Explicit garbage collection
    return data


def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


def generate():
    args = parse_args()
    set_seed_everywhere(args.seed)

    # Load the dataset into shared memory using multiprocessing.Manager
    if not args.prompt_data_field is None:
        df = pd.read_json(
            args.name_or_path,
            lines=True,
            dtype={args.prompt_data_field: "string", args.code_data_field: "string"},
        )
    else:
        df = pd.read_json(
            args.name_or_path,
            lines=True,
            dtype={args.code_data_field: "string"},
        )

    try:
        args.num_samples = int(args.num_samples)
        samples = np.random.choice(
            np.arange(len(df)), size=(args.num_samples,), replace=False
        )
    except ValueError:
        samples = np.arange(len(df))

    # Convert sample indices to numpy array
    samples = np.array(samples)

    # Number of parallel tasks
    num_proc = min(
        args.num_workers, multiprocessing.cpu_count()
    )  # Optimize based on CPU count
    num_paths_per_proc = 8  # Reduced paths to limit memory usage

    # Efficient task distribution across processes
    task_args = [
        (i, min(num_paths_per_proc, len(samples) - i))
        for i in range(0, len(samples), num_paths_per_proc)
    ]

    total_tasks = len(samples) * args.num_edit_paths_per_sample
    batch_size = 64  # Slightly larger batch for faster processing but manageable memory

    with tqdm(total=total_tasks, desc="Processing", ncols=100) as pbar:
        with jsonlines.open(args.dest, mode="w") as writer:
            for batch_start in range(0, len(task_args), batch_size):
                # log_memory_usage()

                # Parallel processing of subprocesses with minimal data passed
                results = Parallel(n_jobs=num_proc, backend="loky", timeout=1000)(
                    delayed(subprocess_task)(
                        start_i,
                        num_samples,
                        args,
                        df,
                        samples,
                    )
                    for start_i, num_samples in task_args[
                        batch_start : batch_start + batch_size
                    ]
                )

                # Collect results and write to file
                for result in results:
                    if result:
                        writer.write_all(result)
                        pbar.update(len(result))

                # log_memory_usage()

                # Explicit garbage collection after processing batch
                gc.collect()
