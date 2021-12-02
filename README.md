# merFISH Experiment debugging Reports

## Installation

    conda env create -f environment.yml

## Usage

The `config.json` file contains the parameters required to run this pipeline. There is no schema in place to validate the inputs so they will be described here:

    "raw_data_path":[str],
    "results_path":[str],
    "raw_image_format":"{wv}nm, Raw/merFISH_{ir}_{fov}_{z}.TIFF"
    "channel": [list of str],
    "fov": [list of str],
    "ir": [list of str],
    "z": [listof str],
    "isRemote":["Y" or "N"]

If `"channel"`, `"fov"`, `"ir"`, and/or `"z"` are empty, they will be auto populated. Currently the `"raw_image_format"` is considered a constant and should not be changed. If isRemote is "Y", then it is assumed that `"raw_data_path"` is a url for a blob with an SAS using the `azcopy` tool.

Once parameters have been selected, the pipeline can be run using

    conda activate experiment_report
    snakemake --cores all --latency-wait 500

## Report Output

Currently the pipeline produces a brightness report and a focus report.

## TODO

1. Explain the reports
2. Add flexability to the `"raw_image_format"`
3. Add schema
