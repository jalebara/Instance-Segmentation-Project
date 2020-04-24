#!/bin/bash

ProjectDir=/home/jabaraho/coding/ECE542FinalProject
HomeDir=/home/jabaraho

read -s -p "Enter password: " sudoPW
trap ' [ "$(ls -A $ProjectDir/data)" ] && echo $sudoPW | umount $ProjectDir/data ; exit 1 ' SIGTERM SIGKILL SIGINT

echo $sudoPW |sudo -S apt update

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
            echo $sudoPW | sudo -S mount -t tmpfs -o size=13G tmpfs $ProjectDir/data
            
            unzip $HomeDir/Downloads/leftImg8bit_trainvaltest.zip -d $ProjectDir/data || exit 1
            mv -f $ProjectDir/data/leftImg8bit/* $ProjectDir/data/  || exit 1
            rm -rf $ProjectDir/data/leftImg8bit || exit 1
            rm $ProjectDir/data/license.txt $ProjectDir/data/README

            unzip $HomeDir/Downloads/gtFine_trainvaltest.zip -d $ProjectDir/data || exit 1
            rsync --remove-source-files --update -a $ProjectDir/data/gtFine/*  $ProjectDir/data/ || exit 1
            rm -rf $ProjectDir/data/gtFine || exit 1
        fi
        python ./experiment.py
    fi
done

if [ "$(ls -A $ProjectDir/data)" ]
then 
    echo "Releasing Ramfs"
    echo $sudoPW | sudo -S umount $ProjectDir/data
fi

echo "Giving up"
