# ESMFold on Snellius HPC

## Introduction
This document provides a guide for setting up and running ESMFold on the Snellius High Performance Computing (HPC) environment. It covers the process of installing necessary dependencies, preparing input FASTA files, and submitting jobs using Slurm.

### Table of Contents
1. [Installation and Environment Setup](#installation-and-environment-setup)
2. [Preparing Your FASTA Files](#preparing-your-fasta-files)
3. [Running ESMFold](#running-esmfold)
4. [Slurm Job Submission Example](#slurm-job-submission-example)

---

## Installation and Environment Setup

### Python Virtual Environment and Dependencies
1. **Create a Directory for ESMFold**:
   ```
   mkdir esmfold
   ```
   
2. **Load Required Modules**:
   Load the Python module and create a virtual environment to manage your Python packages.
   ```
   module load 2022
   module load Python/3.10.4-GCCcore-11.3.0
   python -m venv venv
   ```
   
3. **Activate the Virtual Environment and Install Dependencies**:
   After activating the virtual environment, install PyTorch, torchvision, and other required libraries.
   ```
   module purge
   source venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install transformers py3Dmol accelerate
   ```

   Get the script for running ESMFold.

   ```
   git clone https://github.com/sara-nl/ESMFold_Snellius.git
   cd ESMFold_Snellius
   ```

### Additional Setup Notes
- ESMFold requires CUDA-compatible GPUs for efficient computation. Ensure that the Slurm partition you choose has GPU resources available.
- The installation steps assume the use of CUDA 11.8 and cuDNN 8.6.0.163 modules. Adjust these as necessary based on the available resources on Snellius.

---

## Preparing Your FASTA Files

Ensure your FASTA files are properly formatted before running ESMFold. Each sequence in a FASTA file should begin with a header line starting with ">", followed by the sequence lines. You can organize multiple FASTA files within a designated folder for batch processing. Fasta files should end with ".fa" or ".fasta". 

For instance:

```
7XTB_5|Chain E[auth R]|Soluble cytochrome b562,5-hydroxytryptamine receptor 6|Homo sapiens (9606)
DYKDDDDAKLQTMHHHHHHHHHHHHHHHADLEDNWETLNDNLKVIEKADNAAQVKDALTKMRAAALDAQKATPPKLEDKSPDSPEMKDFRHGFDILVGQIDDALKLANEGKVKEAQAAAEQLKTTRNAYIQKYLASENLYFQGGTVPEPGPTANSTPAWGAGPPSAPGGSGWVAAALCVVIALTAAANSLLIALICTQPALRNTSNFFLVSLFTSDLMVGLVVMPPAMLNALYGRWVLARGLCLLWTAFDVMCCSASILNLCLISLDRYLLILSPLRYKLRMTPLRALALVLGAWSLAALASFLPLLLGWHELGHARPPVPGQCRLLASLPFVLVASGLTFFLPSGAICFTYCRILLAARKQAVQVASLTTGMASQASETLQVPRTPRPGVESADSRRLATKHSRKALKASLTLGILLGMFFVTWLPFFVANIVQAVCDCISPGLFDVLTWLGYCNSTMNPIIYPLFMRDFKRALGRFLPCPRCPRERQASLASPSLRTSHSGPRPGLSLQQVLPLPLPPDSDSDSDAGSGGSSGLRLTAQLLLPGEATQDPPLPTRAAAAVNFFNIDPAEPELRPHPLGIPTN
```

For multimer:

```
>seq1 Human hemoglobin alpha chain
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH
>seq2 Mouse hemoglobin beta chain
MVHLTDAEKAAVSCLWGKVNSDEVGGEALGRLLVVYPWTQRFFASFGNLSSPTAILGNPMVRAHGKKVLTSFGDAVKNLDNIKNTFSQLSELHCDKLHVDPENFRLLGNMIVIVLGHHLGKDFTPAAQAAFQKVVAGVATALAHKYH
>seq3 Yeast cytochrome c
MSSKIQLGYQPKDEGDNGEIKFYVQATCATQIVESNGDVIYDPFDVYHNGYIKHAGGKSAEERKLYSGAQTDGKKVSLSSYVNDGKISRLDAKELGNGCFEFYHKSFKNKDNWIVKLYWQNIDGSFKDGNFKNLRGKTLGAGYVTKGRQTPKHIDNHSFKV
```

---

## Running ESMFold

### Basic Usage
To run ESMFold on your FASTA files, execute the `esmfold.py` script with the paths to your input and output directories specified. Here's a basic command:

```
python esmfold.py --fastas_folder ./outputs/fastas/ --output_folder ./outputs/esmfold/
```

### Script Details
The `esmfold.py` script processes each FASTA file in the input folder, predicting protein structures and saving the results in PDB format in the output folder. It automatically handles the prediction for both monomers and potentially multimers, although specific support for multimers and further optimizations are subject to future updates.

---

## Slurm Job Submission Example

Below is an example of a Slurm job script for running ESMFold. Adjust the resource allocations based on the complexity of your tasks and available resources on Snellius.

```bash
#!/bin/bash
#SBATCH --job-name=esmfold_job
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Activate environment and load necessary modules
source ../venv/bin/activate
module load 2022
module load cuDNN/8.6.0.163-CUDA-11.8.0
cd ./esmfold/

# Define command arguments
cmd_args="--fastas_folder ./outputs/proteinmpnn/seqs/ \
--output_folder ./outputs/esmfold/"

# Run ESMFold
python esmfold.py ${cmd_args}
```

Submit this job script using `sbatch`:

```
sbatch <script_name>.sh
```

---

## Additional Notes
- If you encounter any issues or need further assistance, consider reaching out to Snellius support or consult the documentation for the specific modules and tools you're using.
- Additionally, feel free to reach out to the high performance machine learning team (primary contact: bryan.cardenasguevara@surf.nl) for further assistance.
- For more detailed information about ESMFold parameters and options, refer to the [Transformers documentation by Hugging Face](https://huggingface.co/facebook/esmfold_v1).


---