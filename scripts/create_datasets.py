"""
This script creates processed datasets from raw data files in

`lda/data/[data_name]/`

The raw data files required for each dataset are:

- [data_name]_data.csv which contains a table of [y, X] values without missing data
- [data_name]_helper.csv which contains metadata for each column in [data_name]_data.csv
"""
from lda.paths import get_processed_data_file, get_raw_data_file
from lda.ext.data import BinaryClassificationDataset
from lda.ext.cv import generate_cvindices
import numpy as np

settings = {
    'random_seed': 2338,
    'all_data_names': ['heart', 'adult', 'german', 'fico'],
    }

for data_name in settings['all_data_names']:

    data_file_processed = get_processed_data_file(data_name)

    # create a dataset object by reading a CSV from disk
    data = BinaryClassificationDataset.read_csv(data_file = get_raw_data_file(data_name))

    # generate indices for stratified cross-validation
    strata = (data.y[:, None], data.group_encoder.to_indices(data.G)[:, None])
    strata = np.concatenate(strata, axis = 1)
    _, strata = np.unique(strata, axis = 0, return_inverse = True)
    data.cvindices = generate_cvindices(strata = strata,
                                        total_folds_for_cv=[1, 3, 4, 5],
                                        total_folds_for_inner_cv=[],
                                        replicates=3,
                                        seed=settings['random_seed'])

    data.save(file = data_file_processed, overwrite = True, check_save = True)
    print(f"Processed data saved to {data_file_processed}")

    # data_file_cv_record = get_cv_record_file(data_name)
    # Iterating through each key-value pair in the cvindices dictionary
    # df = data.df
    # for key, value in data.cvindices.items():
    #     df[key] = value
    # # save cv record file
    # df.to_csv(data_file_cv_record, index = False)


