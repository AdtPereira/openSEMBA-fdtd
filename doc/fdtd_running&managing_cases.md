# Running and Managing openSEMBA-fdtd Cases

This document explains the structure and logic of the custom Python script developed to run openSEMBA-FDTD simulations and manage the generated output files.

---

## Project Structure

```
fdtd/
|├── examples/
|   └── holland1981.py        # Python script to run the holland1981 case
|├── examplesData/
|    |
|    ├── cases/                 # Contains .fdtd.json case input files
|    ├── outputs/               # Will store generated .dat and .h5 output files
|    ├── logs/                  # Will store generated .txt report/log files
|    └── excitations/           # Contains excitation (.exc) files
|└── build/bin/semba-fdtd.exe   # Compiled openSEMBA-fdtd executable
```

---

## Script Overview (`examples/holland1981.py`)

This script performs the following:

1. **Prepare the Environment**
   - Insert `src_pyWrapper` and other modules into the Python path.

2. **Copy Excitation File**
   - Copy the excitation file `holland.exc` from `examplesData/excitations/` to the current working directory (`fdtd/`) where the executable runs.

3. **Run the Simulation**
   - Call the `semba-fdtd.exe` solver using `subprocess` with the selected input `.fdtd.json` case.

4. **Organize Outputs**
   - Move all `.dat` and `.h5` files from `examplesData/cases/` into `examplesData/outputs/`.
   - Move all `.txt` files (logs/reports) from `examplesData/cases/` into `examplesData/logs/`.

5. **Cleanup**
   - Remove the temporary copied `holland.exc` file after the simulation.

6. **Verify Outputs**
   - Check if all expected output files were successfully generated.

---

## Output Verification (`check_output_files`)

The function `check_output_files(case_name, output_folder, expected_files)` automatically verifies whether all expected output files exist after the simulation.

### Example Usage:

```python
def check_output_files(case_name, output_folder, expected_files):
    missing_files = []
    found_files = []

    for expected_file in expected_files:
        full_path = os.path.join(output_folder, expected_file)
        if os.path.isfile(full_path):
            found_files.append(expected_file)
        else:
            missing_files.append(expected_file)

    print("\n\ud83d\udcc2 Output file check:")
    if found_files:
        print("\u2705 Files found:")
        for file in found_files:
            print(f"   - {file}")
    if missing_files:
        print("\u274c Files missing:")
        for file in missing_files:
            print(f"   - {file}")
    else:
        print("\n\ud83c\udfaf All expected files found!")
```

### Example Expected Files for holland1981:

```python
expected_files = [
    'holland1981.fdtd_mid_point_Wz_11_11_12_s2.dat',
    'holland1981.fdtd_Energy.dat'
]
```

---

## Notes

- The excitation file must be present in the folder where the solver is run (root `fdtd/` directory).
- The `examplesData/outputs/` and `examplesData/logs/` folders are automatically created if they do not exist.
- If any output file is missing after the simulation, a clear message will be printed.

---

## Future Improvements (Optional)

- Generalize the script to handle any case (not only `holland1981`).
- Automatically detect the required excitation files by parsing the `.fdtd.json` input.
- Automate post-processing (e.g., plotting fields, calculating energy norms).
- Introduce error handling if simulation fails midway.

---

# Summary

This scripting structure ensures:
- Reliable execution of openSEMBA-FDTD cases.
- Proper file organization.
- Automated verification of simulation outputs.
- Clean and maintainable project directories.

---

> Prepared for openSEMBA-FDTD project simulations by ChatGPT 🤖

