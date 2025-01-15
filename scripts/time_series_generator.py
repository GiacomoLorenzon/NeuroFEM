#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# Refactor output filenames (.h5, .xdmf) to generate a time series for paraview.
#
# ------------------------------------------------------------------------------
# Author: Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################


import os
import re


def rename_files(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".xdmf") or filename.endswith(".h5"):
            # Separate the filename and its extension
            name, extension = os.path.splitext(filename)
            # Remove dots between numbers in the name part
            new_name = re.sub(r"(\d+)\.(\d+)", r"\1\2", name)
            # Construct the new filename with the preserved extension
            new_filename = new_name + extension
            # Rename the file
            os.rename(
                os.path.join(directory_path, filename),
                os.path.join(directory_path, new_filename),
            )


def modify_xdmf_file(file_path):
    # Read the content of the XDMF file
    with open(file_path, "r") as file:
        xdmf_content = file.read()

    # Replace float notation in the XDMF file content
    modified_content = re.sub(r"(\d+)\.(\d+)\.h5", r"\1\2.h5", xdmf_content)

    # Write the modified content back to the XDMF file
    with open(file_path, "w") as file:
        file.write(modified_content)


# Example Usage:
directory_path = "."

# Rename both XDMF and HDF5 files in the specified directory
rename_files(directory_path)
print("Files renamed.")

# Iterate through XDMF files in the directory and modify them
for filename in os.listdir(directory_path):
    if filename.endswith(".xdmf"):
        xdmf_file_path = os.path.join(directory_path, filename)
        modify_xdmf_file(xdmf_file_path)
print("xdmf files updated.")
