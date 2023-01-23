#!/bin/bash

for config_file in $(find -path "*config*/*" -name "*.yaml")
do
    echo "Executing config: " $(basename $config_file .yaml) 
    python main.py --config $(basename $config_file .yaml)
done