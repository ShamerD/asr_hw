# ASR project barebones

## Installation guide

Install libraries

```shell
pip install -r ./requirements.txt

git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install 
```

```ctcdecode``` can be somewhat painful to build on Windows (issues with compiling C++ code).
However, on Linux there should be no problems.

Download resources (in project directory)

```shell
wget -q -O ./lm.arpa.gz http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz
gunzip ./lm.arpa.gz

# load model from gdrive in your convenient way (manually or wget)
```

## Before submitting

0) Make sure your projects run on a new machine after complemeting installation guide
1) Search project for `# TODO: your code here` and implement missing functionality
2) Make sure all tests work without errors
   ```shell
   python -m unittest discover hw_asr/tests
   ```
3) Make sure `test.py` works fine and works as expected. You should create files `default_test_config.json` and your
   installation guide should download your model checpoint and configs in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.
   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```
4) Use `train.py` for training

## Notes

Unit tests can be broken in Datasphere due to PermissionErrors.
Colab worked for me as expected (OK)

## Credits

this repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.