# Active Feature Acquisition - Stream

This repository holds the python code for a paper relating to active feature acquisition on streams. It is an extension of work done by Elson Serrao found under this link: https://github.com/elrasp/osm

## Structure of the Repository

additions - additional code not part of framework that was used for the experiments and plots

data - the preprocessed datasets that were used for the runs

osm - core code

requirements.txt - project requirements

## Datasets

The datasets used in the corresponding paper can be found on the UCI website http://archive.ics.uci.edu/ml/datasets.html

Preprocessed versions of them are found under data/csv/

The adult and intrusion data sets have their raw_data.pkl.gzip files further zipped and have to be unpacked to be used

## Running the code

Run code by creating a Framework class in osm/data_streams/algorithm/framework.py and executing process_data_stream()

If unclear may use additions/data_prepare as guideline how to setup a run

The permutations used to produce the paper results can be found within data/_dataset_/prepared as zip files

## Where is what

Framework Class, entry point: osm/data_streams/algorithm/framework.py

Active Feature Acquisition: osm/data_streams/active_feature_acquisition/

Active Learner: osm/data_streams/active_learner/

AFA budget managers; not used by active learners: osm/data_streams/budget_manager/

Oracle: osm/data_streams/oracle/

Windows: osm/data_streams/windows/
