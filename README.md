# ASR project barebones

## Installation guide

Install libraries

```shell
pip install -r ./requirements.txt

git clone --recursive https://github.com/parlance/ctcdecode.git
pip install ./ctcdecode
```

```ctcdecode``` can be somewhat painful to build on Windows (issues with compiling C++ code).
However, on Linux there should be no problems.

Download resources (in project directory)

```shell
wget -q -O ./lm.arpa.gz http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz
gunzip ./lm.arpa.gz

# load model from gdrive in your convenient way (manually or wget)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1I_megQTMRjX51cAK1bihIQ8IRQcECZSs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1I_megQTMRjX51cAK1bihIQ8IRQcECZSs" -O default_test/checkpoint.pth && rm -rf /tmp/cookies.txt
```

## Notes

Unit tests can be broken in Datasphere due to PermissionErrors.
Colab worked for me as expected (OK)

## Credits

this repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.