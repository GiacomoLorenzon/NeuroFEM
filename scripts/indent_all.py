#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# Helper script that uniforms the indentation style all over the project.
#
# Please run it always before pushing into the remote repository or before
# merging with the main branch. This helps keeping track of all modifications
# within the project history.
#
# Usage. Run this script from this folder by typing:
# python3 indent_all.py
#
# ------------------------------------------------------------------------------
# Author: Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################

import os
import subprocess


def format_project(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Formatting {file_path}...")
                try:
                    # Specify the maximum line length (80 characters) using the --line-length option
                    subprocess.run(["black", "--line-length", "80", file_path])
                except Exception as e:
                    print(f"Error formatting {file_path}: {str(e)}")


if __name__ == "__main__":
    project_directory = "../."
    format_project(project_directory)
