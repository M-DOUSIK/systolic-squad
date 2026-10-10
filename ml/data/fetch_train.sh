#!/bin/bash
cd "$(dirname "$0")/train_raw"
for i in 00 01 02 03 04 05 06 07 08 09 10 11; do
  curl -m 900 -L -s -r 0-125829119 -o part$i.tar https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/train-0000$i.tar
  mkdir -p s$i && (cd s$i && tar xf ../part$i.tar 2>/dev/null; true)
  rm -f part$i.tar
done
echo done > ../train_fetch_done
