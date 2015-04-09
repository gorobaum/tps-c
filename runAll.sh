#!/bin/bash

for config in $(cat tps.configs) 
  do 
    ./tps $config >> saida.txt
  done