> 🚨 **Notice (April 2025): This repository has been moved!**
>  
> Please visit the new repository here:  
> 👉 [https://github.com/MarginalCentrality/DieRouterPlus](https://github.com/MarginalCentrality/DieRouterPlus)



# DieRouter+ Experiment Guide  

## Step 1: Create and Activate the Conda Environment  

Create the environment using the provided YAML file:  

```shell
conda env create -f environment.yml
```

Then, activate the environment:  

```shell
conda activate FPGADieRouting
```

**Note:** Ensure to obtain a license of [Mosek](https://www.mosek.com/).  

---

## Step 2: Prepare the Dataset  

1. Create a `Data` directory inside the `DieRouterPlus` directory.  
2. Download the benchmark test cases from [this link](https://edaoss.icisc.cn/file/eventDocuments/sierxinsaishuju.zip).  
3. Extract the downloaded files into the `Data` directory. The directory structure should be as follows:  

```
DieRouterPlus/
├── Data/
│   ├── testcase1/
│   ├── testcase2/
│   ├── testcase3/
│   ├── ...
│   ├── testcase10/
├── Source/
│   ├── Baseline/
│   ├── Baseline_Scripts/
│   ├── ContTDMOPt/
│   ├── ...
│   ├── utils/
├── run_DieRouter+.sh
├── run_DieRouter.sh
```

---

## Step 3: Run the Experiment  

The steps to run **DieRouter+** are encapsulated in the shell script `run_DieRouter+.sh`. Before executing, carefully review the script to understand the hyperparameter settings.  

By default, running the script will process test cases **1-5**, which complete quickly:  

```shell
./run_DieRouter+.sh
```

If you wish to test larger cases (**6-10**), modify the test case range inside `run_DieRouter+.sh` accordingly. 

The experiment results will be stored under the DieRouterPlus/Res directory. For example, when running testcase 1, a corresponding subdirectory testcase1 will be created. Inside this folder, four nested directories will be generated, each representing a specific stage of the routing process. These directories are named using the format run_YYYYMMDDHHMMSS, indicating the timestamp of the run.

A typical experiment run of DieRouter+ produces the following structure:

- testcase1/run_1: stores the initial solution;

- testcase1/run_1/run_2: contains results after rip-up and rerouting based on run_1;

- testcase1/run_1/run_2/run_3: contains the continuous initialization based on run_2;

- testcase1/run_1/run_2/run_3/run_4: stores the final legalized solution based on run_3.

Note: If the program detects that the results of a particular stage under certain hyperparameters already exist, it will automatically reuse the existing directory and proceed to the next stage, avoiding redundant computation. The token parameter in our implementation is used to distinguish different stages and hyperparameter settings.
