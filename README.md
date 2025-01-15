# NeuroFEM

## Installation

These instructions are for Ubuntu 20.04, 

#### Prerequisites
python3 (version >= 3.8) and the following modules:

```bash
pip install pyrameters
pip install -U numpy
pip install pandas
pip install scipy
pip install meshio
```

#### Install FEniCS (version 2019.0.1)
Follow the instructions at the following link:
https://fenicsproject.org/download/archive/.

In particular, the section *Ubuntu FEniCS on Ubuntu*.

## Run a simulation

1. Move to a directory inside `apps`.
The main file is `Problem_Solver.py`.
The help menu can be obtained by
```bash
python3 Problem_Solver.py -h
```
2. Generate the parameter file (with a `Convergence Test` subsection)
```bash
python3 Problem_Solver.py -g param.prm -c 1
```
3. Modify the parameter file `param.prm`
4. Run the simulation (with 1 convergence iteration)
```bash
python3 Problem_Solver.py -f param.prm -c 1
```

## Code style and conventions
Please refer to the following standard coding conventions at the following link:
https://google.github.io/styleguide/pyguide.html.

### Setup your environment
Download the `google_python_style.vim` file if you are coding in vim. Otherwise, rely on Black python interpreter. For instance, if you are using VSCode, download `Black Formatter` extension. Now create, if not already present, the file `pyproject.toml`. Inside:
```
[tool.black]
line-length = 80
```
This fixes the maximum line length to 80 charcters.

Then go into your `.vscode\settings.json` file and add the following lines:
```
"[python]":
{
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "python.formatting.provider": "none"
}
```
### Exploit our scripts
Before committing to the remote repo and before merging, recall to move to `scripts` folder and run `indent_all.py` file. This will properly indent all the files in the project. You can do this by typing:
```
python3 indent_all.py
```
This is fundamental to track the progress among commits and branches.