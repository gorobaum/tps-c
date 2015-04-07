#!/bin/bash

for config in $(cat tps.confs) 
  do 
    ./tps $config >> saida.txt
  done