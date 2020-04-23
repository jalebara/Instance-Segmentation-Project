#!/bin/bash

ProjectDir=/home/jabaraho/coding/ECE542FinalProject

cd $ProjectDir

for i in {1..12}
do
    if [ pgrep -x python >> /dev/null ]
    then
        echo "Computer Still in Use"
        sleep(3600)
    else
        echo "Ready for 5 epoch run"
        conda activate ece542projects

        if [ "$(ls -A /path/to/directory)" ]
        then
            python ./training.py --train_from_checkpoint
        else
            python ./training.py --train_model
        fi
    fi
done

