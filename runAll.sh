#!/bin/bash

for i in `seq 1 10`;
  do 
    ./tps tps.configs >> "saida"$i".txt"
  done