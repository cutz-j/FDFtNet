#!/bin/bash

wget -O StyleGAN_256.tar.001 --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EV99-kK31vhPgd_zwLfgRH8B1MpfPaKKzGILlmfnsb_QOg?download=1

wget -O StyleGAN_256.tar.002 --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EeeOxSYu_MhMglSZ4mIO1HQBbnSvuDewPj34j-kwnr9V3Q?download=1

tar -C ./StyleGAN_256
cat StyleGAN_256.tar* | tar xvf -

rm StyleGAN_256.tar.001
rm StyleGAN_256.tar.002
