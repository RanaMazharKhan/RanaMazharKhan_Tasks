#!/bin/bash
# Task 2 - Linux commands practice

pwd
ls
ls -l

mkdir -p navttc_practice
cd navttc_practice
touch demo.txt
echo "NAVTTC Linux practice" > demo.txt
echo "Second line" >> demo.txt

cat demo.txt
head -n 1 demo.txt

cp demo.txt copy.txt
mv copy.txt renamed.txt
ls

file demo.txt
ls -l demo.txt
chmod 644 demo.txt

ps aux | head -5

# man ls            (manual page, press q to quit)
# sudo dpkg -l      (list installed packages)
# sudo shutdown now (powers off - explanation only, not run)

cd ..
rm -r navttc_practice
echo "done"
