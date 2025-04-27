
# OpenSEMBA-FDTD Installation and Compilation Guide (Windows)

This step-by-step guide shows how to prepare your environment, compile, and run the OpenSEMBA FDTD solver.

## 1. Prerequisites

Before starting, make sure you have:

1. **Git** (to clone the repository)
   - Download from https://git-scm.com/downloads and install it.

2. **Python 3.8+**
   - Recommended: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight and easy to use).
   - During installation, check **Add to PATH**.

3. **CMake** and **Ninja**
   - Open a terminal (PowerShell or CMD) and install via Conda:
     ```powershell
     conda install -c conda-forge cmake ninja
     ```
   - Or download installers from https://cmake.org/download/ and https://github.com/ninja-build/ninja/releases.

4. **Microsoft Visual Studio 2019 or 2022**
   
   - Visual Studio Community Edition is free for students, open-source developers, and individual use. You can obtain Microsoft Visual Studio 2019 or Visual Studio 2022 officially from Microsoft's website:

      - Visual Studio 2022: https://visualstudio.microsoft.com/vs/
      - Visual Studio 2019 (older versions / archive): https://learn.microsoft.com/en-us/visualstudio/releases/2019/history

   - Install Visual Studio with C++ support. During the installation, enable:
      - `Desktop development with C++`
      - `C++ Build Tools`

## 2. Fortran Compiler

   This section documents the installation of Intel toolkits and the usage of the Intel Fortran Compiler (`ifx`) directly on Windows.

   - **Download Intel toolkits:**

      Access and install from the following links:
      - [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
      - [Intel® oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html)

      Make sure to include the **Intel Fortran Compiler (`ifx`)** during installation.

   - **Configure the environment in the terminal**

      After installation, open the Command Prompt and run:

      ```cmd
      cd C:\Program Files (x86)\Intel\oneAPI\
      setvars.bat
      ```

      This will automatically configure the necessary environment variables.

      ```cmd
      C:\Program Files (x86)\Intel\oneAPI>setvars.bat
      :: initializing oneAPI environment...
         Initializing Visual Studio command-line environment...
         Visual Studio version 17.13.6 environment configured.
         "C:\Program Files\Microsoft Visual Studio\2022\Community\"
         Visual Studio command-line environment initialized for: 'x64'
      :  compiler -- latest
      :  debugger -- latest
      :  dev-utilities -- latest
      :  dpcpp-ct -- latest
      :  dpl -- latest
      :  ipp -- latest
      :  ippcp -- latest
      :  mkl -- latest
      :  mpi -- latest
      :  ocloc -- latest
      :  pti -- latest
      :  tbb -- latest
      :  umf -- latest
      :: oneAPI environment initialized ::
      ```

   - **Verify the installation**

      In the same terminal, type:

      ```cmd
      where ifx
      ```

## 3. Clone the repository

Open your terminal and run:

```bash
git clone https://github.com/OpenSEMBA/fdtd.git
cd fdtd
```

This will create the `fdtd` folder with all the source code.

## 4. Initialize external submodules

The project uses external libraries under `external/`. Fetch them with:

```bash
git submodule update --init --recursive
```

## 5. Create the Python environment

- Create and activate a Conda environment:
   ```powershell
   conda create -n semba-fdtd python=3.10
   conda activate semba-fdtd
   ```
- Install the Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## 6. Configure and compile with CMake + Ninja

Go to your root directory (`fdtd`):

``` cmd
cd C:\Users\...\openSEMBA\fdtd
```

Then, run:

```bash
# specify the Fortran compiler (ifx)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=ifx

# build everything
cmake --build build
```

After compilation, the executable will be located at `build/bin/semba-fdtd(.exe)`.

## 7. Integration with VS Code (Optional)

1. Create `.vscode/settings.json`:
   ```jsonc
   {
     "cmake.generator": "Ninja",
     "cmake.sourceDirectory": "${workspaceFolder}",
     "cmake.buildDirectory": "${workspaceFolder}/build"
   }
   ```
2. Create `.vscode/tasks.json` with build and run tasks:
   ```jsonc
   {
     "version": "2.0.0",
     "tasks": [
       { "label": "build", "command": "cmake --build build", "group": "build" },
       { "label": "run",   "command": "${workspaceFolder}/build/bin/semba-fdtd.exe", "args": ["-i","testData/input_examples/mtln.fdtd.json"] }
     ]
   }
   ```
3. Use **Ctrl+Shift+B** to build and **Terminal ▶ Run Task ▶ run** to execute.

## 8. Troubleshooting Tips

- **Compiler not found**: check with `where ifx` before running CMake.
- **Submodule errors**: ensure you have run `git submodule update --init --recursive`.
- **Spaces in file paths**: use quotes `"..."` or place the project in a path without spaces (e.g., `C:/dev/fdtd`).
- **Python environment**: always activate the Conda environment before using `pip install` and `python`.
