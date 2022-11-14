#!/bin/bash

round=0
until [ $round -ge 50000 ] 
do
   python read_cifar100train.py --line_number $round
   let round+=100
done
let round=0
until [ $round -ge 10000 ] 
do
   python read_cifar100test.py --line_number $round
   let round+=100
done