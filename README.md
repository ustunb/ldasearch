# `ldasearch` 
This library contains Python code to search for a least discriminatory classifier


## Setup

#### Clone the git repository

```
$ git clone git@github.com:ustunb/ldasearch.git 
```

#### Requirements

- Python 3.9+ 
- [CPLEX 22.1.0.0](https://pypi.org/project/cplex/) 


CPLEX is fast optimization solver with a Python API. It is commercial software, but free to download for students and faculty at accredited institutions. To obtain CPLEX:

1. Register for [IBM OnTheHub](https://ur.us-south.cf.appdomain.cloud/a2mt/email-auth)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://www-03.ibm.com/isc/esd/dswdown/searchPartNumber.wss?partNumber=CJ6BPML)
3. Install CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems with CPLEX, please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059).

## Workflow

1. **Run [`scripts/create_datasets.py`](scripts/create_datasets.py)**    
This script will create a dataset object from a CSV file with a particular format and generate cross-validation indices. 

2. **Run [`scripts/train_baseline_model.py`](scripts/train_baseline_model.py)**   
This script will fit a baseline classification model using a processed dataset.

3. **Run `scripts/train_lda_model.py`.**   
This script will search for a least discriminatory alternative for the baseline model.

The Further instructions can be found at the top of each script.

### Directory Structure

```
├── data         # pickle files with processed datasets    `data_dir`
├── lda          # source code for process/training        `pkg_dir`
├── scripts      # scripts to create results                        
└── results      # pickle files with results               `results_dir`
```

## Reference

For more about this method, check out our paper:

> [Operationalizing the Search for Less Discriminatory Alternatives in Fair Lending](https://dl.acm.org/doi/abs/10.1145/3630106.3658912)

If you use this library in your research, we would appreciate a citation!

```
@inproceedings{gmu2024lda,
author = {Gillis, Talia B and Meursault, Vitaly and Ustun, Berk},
title = {Operationalizing the Search for Less Discriminatory Alternatives in Fair Lending},
year = {2024},
isbn = {9798400704505},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3630106.3658912},
doi = {10.1145/3630106.3658912},
pages = {377–387},
numpages = {11},
location = {Rio de Janeiro, Brazil},
series = {FAccT '24}
}
```


If you use this code in your research, please cite the following paper:
