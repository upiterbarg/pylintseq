# pylintseq

A minimal package implementing the [LintSeq algorithm](https://lintseq.github.io/) for Python code, as described in [Piterbarg et al. 2024](https://arxiv.org/abs/2410.02749).

LintSeq reparameterizes code synthesis with language models into a sequential code edit generation problem, by refactoring programs in the training corpuses across equivalent *edit paths*. 

## Installation

To build `lintseq` from source, clone this repository and pip install.

```
git clone https://github.com/upiterbarg/pylintseq.git
cd pylintseq
pip install .
```

## Usage

Once installed in your Python environment, you can run parallelized `pylintseq` from any directory on an LM training corpus of Python programs using a single line of code. The training corpus must be formatted as a JSONLines file.

```
pylintseq \
    -p PATH_TO_JSONLINES_DS \
    -d DESTIONATION_DIR \                                  # default: saves data to the current working directory
    --prompt_data_field NAME_OF_PROMPT_DATA_FIELD \        # default: 'instruction' (pass as 'None' if not defined)
    --code_data_field NAME_OF_CODE_DATA_FIELD \            # default: 'response'
    -s NUMBER_OF_EDIT_PATHS_TO_GENERATE_PER_SAMPLE  \      # default: 1
    -c NUMBER_OF_CORES_TO_USE  \                           # default: 8
    --seed RANDOM_SEED \                                   # default: 1
```

By default, the processed dataset will be generated in the current working directory (as a JSONLines file). To generate it elsewhere, you can specify a different target path by using the arguments `-d` or `--dest_dir`.

To run LintSeq on only a (randomly sampled) subset of your dataset, pass the additional argument `-n` or `--num_samples` with the desired sample count during launch.

## Reading a pylintseq Generated Dataset

`pylintseq` saves processed data to the JSONLines format. You can load it using your favorite JSONLines reader. An example using `pandas` is shown below. 

```
>>> import pandas as pd
>>> df = pd.read_json(PATH_TO_PYLINTSEQ_DS, lines=True)
>>> df
   edit_path                                          index   source_file            source_instruction                                 source_response
0  [@@ -0,0 +1,6 @@\n+import statistics\n+\n+def ...    665   my_code_dataset.jsonl  Write a Python function that takes a list of g...  Here is the implementation:\n\n```python\nimpo...
1  [@@ -0,0 +1,5 @@\n+def get_resource():\n+    r...  63189   my_code_dataset.jsonl  You are tasked with creating a simple web serv...  ```python\nfrom flask import Flask, jsonify\n\...
2  [@@ -0,0 +1,6 @@\n+def create_file_from_templa...  24173   my_code_dataset.jsonl  Write a Python function `create_file_from_temp...  To create a file from a template, you need to ...
3  [@@ -0,0 +1,3 @@\n+def print_pattern(n: int) -...  61605   my_code_dataset.jsonl  You are given a Python code snippet that print...  ```python\ndef print_pattern(n: int) -> None:\...
4  [@@ -0,0 +1,10 @@\n+from models import RoomCom...  60850   my_code_dataset.jsonl  You are tasked with creating a RESTful API end...  ```python\nfrom models import RoomComponent  #...
5  [@@ -0,0 +1 @@\n+# main.py, @@ -1,0 +2,8 @@\n+...  55297   my_code_dataset.jsonl  You are working on a Python project that invol...  ```python\n# main.py\n\nfrom subdirectory impo...
6  [@@ -0,0 +1,3 @@\n+def add_record(lst, records...  20053   my_code_dataset.jsonl  Create a Python function `add_record` that add...  Here's how we can implement the `add_record` f...
7  [@@ -0,0 +1 @@\n+from bs4 import BeautifulSoup...  64421   my_code_dataset.jsonl  You are tasked with creating a Python function...  ```python\nfrom bs4 import BeautifulSoup\n\nde...
8  [@@ -0,0 +1 @@\n+import ast, @@ -0,0 +1 @@\n+f...  55104   my_code_dataset.jsonl  You are tasked with creating a Python function...  ```python\nfrom typing import List\nimport ast...
9  [@@ -0,0 +1,3 @@\n+class NetworkDevice:\n+    ...  55801   my_code_dataset.jsonl  You are tasked with creating a Python class th...  ```python\nfrom netmiko import ConnectHandler\...
```

Edit sequences are saved as lists of strings to a column called `edit_path`. The contents of the data fields `NAME_OF_PROMPT_DATA_FIELD` and `NAME_OF_CODE_DATA_FIELD` in the original dataset will be respectively saved to columns titled `source_${NAME_OF_PROMPT_DATA_FIELD}` and `source_${NAME_OF_CODE_DATA_FIELD}`.

## FAQs

 > **Can I run `pylintseq` on code data that might contain natural language chain-of-thought (CoT) traces?**
 
 >> Yes. If your code data contains any natural lang CoT traces interleaved with Python in Markdown format, these traces will be stripped from data during processing.

> **The dataset I ran `pylintseq` on had *m* examples in it, but some of these examples are missing from the output dataset. Is there a bug in the code?**
>> No. This will occur if there are programs in your dataset that have no executable Python code, e.g. if they consist of comments only. Such examples will be detected and removed from data output by `pylintseq` during processing.

 > **I'm running `pylintseq` on a large dataset and the progress bar is updating slowly. Is `pylintseq` still running?**
 >> This implementation of the LintSeq algorithm is optimized for high throughput and low memory load on large data streams -- data is processed in batches, and updates to the progress bar are similarly batched. To speed up processing, you can increase the number of worker cores on launch using the key word arguments `-c` and `--num_workers`.

## Citation
```
@misc{piterbarg2024editseq,
      title={Training Language Models on Synthetic Edit Sequences Improves Code Synthesis}, 
      author={Ulyana Piterbarg and Lerrel Pinto and Rob Fergus},
      year={2024},
      eprint={2410.02749},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
