#!/bin/bash

ProjectDir=/home/jabaraho/coding/ECE542FinalProject
HomeDir=/home/jabaraho

read -s -p "Enter password" sudoPW
trap ' [ "$(ls -A $ProjectDir/data)" ] && umount $ProjectDir/data ; exit 1 ' SIGTERM SIGKILL SIGINT

echo $sudoPW |sudo apt update

cd $ProjectDir

for i in {1..8}
do
    if pgrep -x python >/dev/null
    then
        echo "Computer Still in Use"
        sleep 3600
    else
        echo "Ready for 5 epoch run"

        if [ "$(ls -A $ProjectDir/data)" ]
        then 
            echo "Data Already Loaded"
        else
            echo "Generate Ramdisk for data"
            echo $sudoPW | sudo mount -t tmpfs -o size=13G tmpfs $ProjectDir/data
            
            unzip $HomeDir/Downloads/leftImg8bit_trainvaltest.zip -d $ProjectDir/data || exit 1
            mv -f $ProjectDir/data/leftImg8bit_trainvaltest/* $ProjectDir/data || exit 1
            rm -rf $ProjectDir/data/leftImg8bit_trainvaltest || exit 1

            unzip $HomeDir/Downloads/gtFine_trainvaltest.zip -d $ProjectDir/data || exit 1
            rsync --remove-source-file -a $ProjectDir/data/gtFine || exit 1
            rm -rf $ProjectDir/data/gtFine || exit 1
        fi

        if [ "$(ls -A $ProjectDir/logs)" ]
        then
            python ./training.py --train_from_checkpoint || { echo "Checkpoint Training Failed... Releasing Ramfs" ; echo $sudoPW | sudo umount $ProjectDir/data ; exit 1 ; }
        else
            python ./training.py --train_model || { echo "Initial Training Failed... Releasing Ramfs" ; echo $sudoPW | sudo umount $ProjectDir/data ;  exit 1 ; }
        fi
    fi
done

if [ "$(ls -A $ProjectDir/data)" ]
then 
    echo "Releasing Ramfs"
    echo $sudoPW | sudo umount $ProjectDir/data
