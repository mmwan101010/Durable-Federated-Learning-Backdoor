#!/bin/bash

round=0
until [ $round -ge 10000 ] 
do
   python read_cifar10.py --line_number $round
   let round+=100
done