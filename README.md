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
```

---

## Step 3: Run the Experiment  

The steps to run **DieRouter+** are encapsulated in the shell script `run.sh`. Before executing, carefully review the script to understand the hyperparameter settings.  

By default, running the script will process test cases **1-5**, which complete quickly:  

```shell
./run.sh
```

If you wish to test larger cases (**6-10**), modify the test case range inside `run.sh` accordingly. 
