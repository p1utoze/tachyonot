#! /bin/bash
printf 'Make sure to copy these files first to the target device:\ntachyonot-cli.dist\nmodel folder\nconfig.yaml\ndeb-pkgs folder\ndocs folder\n'
echo ''
echo 'export CONFIG_PATH=/media/root-rw/config.yaml' >> ~/.bashrc 
source ~/.bashrc 

apt install deb-pkgs/*.deb 

./tachyonot-cli.dist/cli/cli --help

