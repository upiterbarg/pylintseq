import contextlib
import io
import numpy as np
import gzip
import json
import os
import pdb
import random
import tempfile
import difflib
from pylint import run_pylint
from typing import Iterable, Dict
from copy import deepcopy


def set_seed_everywhere(seed):
    """Set random seed everywhere."""
    np.random.seed(seed)
    random.seed(seed)


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io(stream):
    """Capture outputs"""
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield stream


def file_linter(file, expected_error_traces=None):
    """
    Cleanly pylint contents of temp file. Treat identation warnings as errors.

    Return all (unexpected) error traces in standardized template formatting.
    """
    try:
        program_io = io.StringIO()
        with swallow_io(program_io):
            run_pylint(
                (
                    "--disable=C,R",
                    "--reports=n",
                    "--score=n",
                    "--unsafe-load-any-extension=y",
                    "--generated-members=cv2.*",
                    "--msg-template={C}|{msg_id}|{line}|{column}|{msg}",
                    file,
                )
            )
    except BaseException as e:
        pass
    io_stream = program_io.getvalue()
    io_stream = io_stream.split("\n")
    error_traces = []
    for line in io_stream:
        try:
            C, msg_id, line_id, column_id, msg = line.split("|")
        except:
            continue

        # treat indentation warnings as errors
        if C == "W" and msg_id not in ["W0311", "W0231"]:
            continue
        elif msg_id == "E0001" and "on line" in msg:
            line_id = msg[: msg.rfind("(")].rstrip().split(" ")[-1]

        error_trace = (
            msg_id,
            line_id,
            column_id,
            msg,
        )
        if expected_error_traces is None or not error_trace in expected_error_traces:
            error_traces += [error_trace]
    return error_traces


def get_diff(string_one, string_two, n=0):
    """Compute a clean text delta using 'difflib' between two strings.

    Return the text delta as a single string.
    """
    proc_string_one = string_one.strip().splitlines()
    proc_string_two = string_two.strip().splitlines()

    out = "\n".join(
        [
            line
            for i, line in enumerate(
                difflib.unified_diff(
                    proc_string_one,
                    proc_string_two,
                    n=n,
                    fromfile="file1",
                    tofile="file2",
                    lineterm="",
                )
            )
            if i > 1
        ]
    )
    return out


def inflate_edit_path(code_as_text, edit_sequence):
    """Apply the line indices of edits to the string representing the program/file.

    Return the corresponding code edits, expressed both as sequences of "raw" program
    states and as code "diffs".
    """
    lines = code_as_text.split("\n")

    integrated = []
    total_edit = []

    for step, edit in enumerate(edit_sequence):
        total_edit += edit
        integrated += [deepcopy(total_edit)]

    raw_text_seq = [code_as_text]
    for edit in integrated:
        out = "\n".join([line for (i, line) in enumerate(lines) if not i in edit])
        raw_text_seq += [out]

    del integrated
    del total_edit

    if raw_text_seq[-1] != "":
        raw_text_seq += [""]

    raw_text_seq = raw_text_seq[::-1]

    diff_text_seq = []

    for i in range(len(raw_text_seq) - 1):
        diff_text_seq += [get_diff(raw_text_seq[i], raw_text_seq[i + 1])]
    return raw_text_seq, [d for d in diff_text_seq if len(d) > 0]


def strip_chain_of_thought(response, expected_programming_language="python"):
    """Given an input example "response", strip away any chain-of-thought-like natural language
    by looking for Markdown formatting.

    Return the resultant response.
    """
    suffix = response
    code_chunks = []
    while f"```{expected_programming_language}" in suffix:
        code_chunk_suffix = suffix[
            suffix.find(f"```{expected_programming_language}")
            + len(f"```{expected_programming_language}") :
        ]
        code_chunks += [code_chunk_suffix[: code_chunk_suffix.find("```")]]
        suffix = code_chunk_suffix[code_chunk_suffix.find("```") + len("```") :]

    # merge parsed code chunks into a single string
    if len(code_chunks) > 0:
        out = "\n".join(code_chunks)
        return out.lstrip().rstrip()
    else:
        return response


def lintseq_backward_sampling_pythonic(
    code_as_text: str,
    children_per_round: int = 16,
    top_k: int = 4,
    max_population_size: int = 64,
    max_depth: int = 1,
    indent_bias_sampling_factor=None,
    ignore_imports: bool = False,
    ignore_comments: bool = True,
    ignore_global_defs: bool = True,
    ignore_init_errors: bool = False,
) -> list:
    """Implements the backward sampling phase of the LintSeq algorithm: given a text string
    representing a program, search over the space of possible sequences of error free insertion
    edits that can be used to write the program line by line by sampling deletions. This
    implementation of the algorithm supports a few additional hyperparameters (inspired by
    evolutionary search) beyond the default version of the algorithm introduced by Piterbarg
    et al 2024. Each call expands a search tree over possible synthetic edit sequences.


    Keyword arguments:
        > code_as_text -- a string representing a program
        > children_per_round -- the maximal number of "child edits" to expand from each "branch"
            during each round of sampling
        > top_k -- the number of "child edits" per branch to keep for the next round
        > max_population size -- the total maximal number of edit sequences ("leaves") to preserve
        > max_depth -- the maximal depth (i.e. number of edits) of any edit sequence in the tree
        > indent_bias_sampling_factor -- if passed as a positive number, biases sampling of edit
            deletions by indentation accordingly
        > ignore_imports -- should import statements be included in edits?
        > ignore_comments -- should comments be included in edits?
        > ignore_global_defs -- should global vars be included in edits?
        > ignore_init_errors -- should any errors that are present in the program before sampling
            start be ignored during edit sampling?

    Returns:
        > A list of all of the sampled edit sequences (i.e. expanded "paths" from root to leaf)
    """

    def _apply_deletion_edit(edit):
        """
        Apply a deletion edit, represented as a list of line indices to be deleted.

        Returns:
            > A string representing the full program post deletion
            > A list of the remaining line indices in the program, post deletion
        """
        return "\n".join(
            [line for i, line in enumerate(code_as_text.split("\n")) if not i in edit]
        ), [i for i in range(len((code_as_text.split("\n")))) if not i in edit]

    def _get_weight_by_line(
        lines,
        candidate_lines,
        indent_bias_sampling_factor=indent_bias_sampling_factor,
    ):
        """Get sampling weights. Supports preferential sampling of code lines with "deeper" indentation
        via the "indent_bias_sampling_factor" parameter.

        Returns:
            > A numpy array of line sampling weights
            > A list of indentations per line (in spaces)
            > The detected tab width (in spaces)
        """
        tab_width = None
        weights = []
        indents = []
        li = 0

        for i, line in enumerate(lines):
            if len(line.lstrip()) > 1:
                li = len(line) - len(line.lstrip())
            indents += [li]
            if tab_width is None and li > 0:
                tab_width = li

            if i in candidate_lines:
                weights += [
                    1
                    if (li == 0 or indent_bias_sampling_factor is None)
                    else (li / tab_width * indent_bias_sampling_factor)
                ]
            else:
                weights += [0]
        return np.array(weights), indents, tab_width

    def _lookup_children(target_line, indents, tab_width):
        """Look up whether a target line for deletion has any dependent children, based
        on indentation.

        If attempting to apply LintSeq to languages other than Python, this method might
        need to be adjusted.

        Returns:
            > A list of line indices that are "dependent" on the target line.
        """
        target_indent = indents[target_line]

        ## no children case
        if target_line == len(indents) - 1 or target_indent >= indents[target_line + 1]:
            return []

        indents_subseq = indents[target_line + 1 :]

        ### larger than 0 --> child, smaller than or equal to zero --> independent
        ### index is 1 if independent, 0 if it's a child
        indices = [int((i - target_indent) <= 0) for i in indents_subseq] + [1]
        return [i for i in range(target_line + 1, indices.index(1) + target_line + 1)]

    def _get_child_edit_induced_deletion(
        edit_candidate,
        previous_edits,
        indents,
        tab_width,
        expected_error_traces=None,
        max_depth=4,
    ):
        """Delete the line with index "edit_candidate" from the program, lint the resultant
        code, gather any induced error traces, add affected lines indices to the
        "deletion" queue, and recurse until there are no errors detected.

        Returns:
            > All deleted line indices
            > The total count of lines dependent on "edit_candidate"

        """
        fail = True
        induced_deletion_size = 0
        depth = 0
        rm = [i for i in range(len(lines))]
        affected_lines = edit_candidate
        edit = []

        while fail:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="r+") as fp:
                if len(affected_lines) > 0:
                    affected_children = []
                    for line in affected_lines:
                        new_children = [
                            c
                            for c in _lookup_children(line, indents, tab_width)
                            if not (c in edit or c in affected_lines)
                        ]
                        affected_children += new_children
                    affected_lines += affected_children

                edit += affected_lines

                edited, rm = _apply_deletion_edit(edit + previous_edits)
                fp.write(edited)
                fp.seek(0)

                error_traces = file_linter(
                    fp.name, expected_error_traces=expected_error_traces
                )

                induced_deletion_size += len(error_traces)
                fail = len(error_traces) > 0

                if fail:
                    affected_lines = [
                        rm[int(line_id) - 1]
                        for (msg_id, line_id, column_id, msg) in error_traces
                    ]

                depth += 1

                fp.close()

        return edit, induced_deletion_size

    # Split lines
    lines = code_as_text.split("\n")

    default_candidate_lines = [i for i in range(len(lines))]

    if ignore_imports:
        default_candidate_lines = [
            i
            for i, line in enumerate(lines)
            if i in default_candidate_lines and not "import" in line
        ]
    if ignore_comments:
        default_candidate_lines = [
            i
            for i, line in enumerate(lines)
            if i in default_candidate_lines
            and len(line.lstrip()) > 0
            and not "#" == line.lstrip()[0]
        ]

    if ignore_global_defs:
        open_global_def = False
        candidate_lines = []
        global_def_type = None
        for i, line in enumerate(lines):
            if not i in default_candidate_lines:
                continue

            if not open_global_def:
                candidate_lines += [i]

            if open_global_def:
                if global_def_type == "parenth" and line.rstrip()[-1] == ")":
                    open_global_def = False
                    global_def_type = None
                elif global_def_type == "square" and line.rstrip()[-1] == "]":
                    open_global_def = False
                    global_def_type = None
            elif (
                "=" in line
                and line[: line.find("=")].replace(" ", "").isupper()
                and (line.rstrip()[-1] in ["(", "["])
            ):
                open_global_def = True
                if line.rstrip()[-1] == "(":
                    global_def_type = "parenth"
                else:
                    global_def_type = "square"

        default_candidate_lines = candidate_lines

    init_errors = None
    if ignore_init_errors:
        candidate_lines_as_text = "\n".join([lines[i] for i in default_candidate_lines])
        with tempfile.NamedTemporaryFile(
            delete_on_close=True, suffix=".py", mode="r+"
        ) as fp:
            fp.write(candidate_lines_as_text)
            fp.seek(0)
            error_traces = file_linter(fp.name, expected_error_traces=None)
            fp.close()
        init_errors = error_traces

    # initialize search state params
    edit_seq_population = []
    edit_seq_candidates = []
    depth = 0

    W, I, tab_width = _get_weight_by_line(lines, default_candidate_lines)
    default_probs = W / W.sum()  ## normalize to yield well-defined probs

    while len(edit_seq_population) < max_population_size and depth < max_depth:
        ### Sample candidate children edits

        #### Base case
        if len(edit_seq_candidates) == 0:
            ## Sample $(children_per_round) total edit(s)
            seeds = np.arange(len(lines))

            try:
                children = (
                    np.random.choice(
                        seeds, (children_per_round,), p=default_probs, replace=False
                    )
                    .astype(int)
                    .tolist()
                )
            except:
                return None
            children = [children]  # No parent
        else:
            children = []
            for _, remaining_lines, _ in edit_seq_candidates:
                dW = W[np.array(remaining_lines)]
                rel_probs = dW / dW.sum()
                children_by_parent = (
                    np.random.choice(
                        remaining_lines,
                        (min(children_per_round, len(remaining_lines)),),
                        p=rel_probs,
                        replace=False,
                    )
                    .astype(int)
                    .tolist()
                )
                children += [children_by_parent]

        ### Compute the induced deletion set of each "child" edit.
        ### Sort these edits in order of their dependency set sizes.
        child_info = []
        for parent_id in range(len(children)):
            children_by_parent = children[parent_id]
            child_info_by_parent = []

            if len(edit_seq_candidates) == 0:
                integrated = []
            else:
                _, remaining_lines, integrated = edit_seq_candidates[parent_id]
            for child in children_by_parent:
                edit, induced_deletion_size = _get_child_edit_induced_deletion(
                    [child],
                    integrated,
                    I,
                    tab_width,
                    expected_error_traces=init_errors,
                )
                child_info_by_parent += [(edit, induced_deletion_size)]
            child_info += [child_info_by_parent]

        ### Update candidates
        #### Top_k supported for now only
        sorted_children = [
            sorted(children_by_parent, key=lambda tup: tup[1])[::-1]
            for children_by_parent in child_info
        ]
        candidate_children = [
            children_by_parent[:top_k] for children_by_parent in sorted_children
        ]

        ## Updates candidates --> fold edit children with their parents
        new_candidates = []

        for parent_id in range(len(candidate_children)):
            children_by_parent = candidate_children[parent_id]

            for child in children_by_parent:
                child_edit, child_induced_deletion = child
                child_edit = list(set(child_edit))

                ## Base case
                if len(edit_seq_candidates) == 0:
                    ### Each candidate consists of:
                    ### (1) compact edit sequence (list of int lists) and
                    ### (2) remaining candidate lines and
                    ### (3) integrated program state
                    new_candidate = (
                        [child_edit],
                        [i for i in default_candidate_lines if not i in child_edit],
                        child_edit,
                    )
                    new_candidates += [new_candidate]
                    continue

                ## Subsequent rounds
                parent = edit_seq_candidates[parent_id]
                ### Unpack parent program info
                edit_sequence, remaining_lines, integrated = parent

                new_candidate = (
                    edit_sequence[:] + [child_edit[:]],
                    [i for i in remaining_lines if not i in child_edit],
                    list(set(integrated + child_edit[:])),
                )
                new_candidates += [new_candidate]

        ### Update population, candidate tracking
        edit_seq_candidates = []
        for candidate in new_candidates:
            edit_sequence, remaining_lines, integrated = candidate
            if len(remaining_lines) > 0:
                edit_seq_candidates += [candidate]
            else:
                edit_seq_population += [(edit_sequence, remaining_lines, integrated)]

        if len(edit_seq_candidates) >= max_population_size and depth < max_depth:
            random.shuffle(edit_seq_candidates)
            edit_seq_candidates = edit_seq_candidates[: max_population_size // top_k]

        ## Update edit sequence depth
        depth += 1

    ### Grow seq population as much as possible
    if len(edit_seq_population) < max_population_size:
        sorted_candidates = sorted(
            edit_seq_candidates, key=lambda tup: len(tup[1])
        )  # Ascending order
        best_candidates = sorted_candidates[
            : min(
                len(sorted_candidates),
                max_population_size - len(edit_seq_population) + 1,
            )
        ]
        edit_seq_population += best_candidates

    # Cut seq population down to max, sorting by coverage
    sorted_edit_seq_population = sorted(
        edit_seq_population, key=lambda tup: len(tup[1])
    )  ### then, by coverage
    edit_seq_population = sorted_edit_seq_population[:max_population_size]

    return edit_seq_population
