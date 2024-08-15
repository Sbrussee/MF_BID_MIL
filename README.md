## MF-CLAM
This is the repository for the CLAM model applied to the task of differentating between Mycosis Fungoides (MF) and Benign Inflammatory Dermatoses (BIDs). The implementation heavily relies on the SlideFlow Python Package (ver 2.3.0).

## Installation
In order to run the script, one can use pip to install the requirements:
```
pip install -r requirements.txt
```


## Inference
To run the code, one can either specify CLI arguments to mf_script.py or use the multi_input.json file to specify the parameters. When using the CLI, one can run the script as follows:

```
python mf_script.py -p /path/to/project_directory \
                    -s /path/to/slide_directory \
                    -a /path/to/annotation_file.csv \
                    -tf 0.1 \
                    -k 10 \
                    -ts 256 \
                    -mg 40x \
                    -ag xyjn \
                    -l patient \
                    -b tile \
                    -f RetCCL \
                    -m Attention_MIL \
                    -n macenko \
                    -sp v3 \
                    -se
```

The following CLI-arguments can be used:
- `-p`, `--project_directory`: Directory to store the project.
- `-s`, `--slide_directory`: Directory where slides are located.
- `-a`, `--annotation_file`: CSV file having the slide id's, labels, and patient id's. It should, at least, contain a 'slide' and 'patient' column.
- `-tf`, `--test_fraction`: Fraction of dataset to hold apart for testing. (Default: 0.1)
- `-k`, `--k_fold`: Number of folds to use for k-fold cross-validation. (Default: 10)
- `-ts`, `--tile_size`: Size of tiles to use in pixels. (Default: 256)
- `-mg`, `--magnification`: Magnification level to use. Choices: ['40x', '20x', '10x', '5x'] (Default: '40x')
- `-ag`, `--augmentation`: Augmentation methods to use. Can be any combination of x: random x-flip, y: random y-flip, r: random cardinal rotation, j: random JPEG compression, b: random gaussian blur, n: stain augmentation. Example: 'xyjn' (Default: 'xyjrbn')
- `-l`, `--aggregation_level`: Level of bag aggregation to use. Choices: ['patient', 'slide']
- `-b`, `--training_balance`: Balances batches for training. Choices: ['tile', 'slide', 'patient', 'category']
- `-f`, `--feature_extractor`: Pretrained feature extractors to use. Choices: ['CTransPath', 'RetCCL', 'HistoSSL', 'PLIP', 'SimCLR', 'DinoV2', 'resnet50_imagenet', 'barlow_twins_feature_extractor'] (Default: 'RetCCL')
- `-m`, `--model`: MIL model to use. Choices: ['Attention_MIL', 'CLAM_SB', 'CLAM_MB', 'MIL_fc', 'MIL_fc_mc', 'TransMIL'] (Default: 'Attention MIL')
- `-n`, `--normalization`: Normalization method to use. Choices: ['macenko', 'vahadane_sklearn', 'reinhard', 'cyclegan', 'None'] (Default: 'macenko')
- `-sp`, `--stain_norm_preset`: Stain normalization preset parameter sets to use. Choices: ['v1', 'v2', 'v3'] (Default: 'v3')
- `-j`, `--json_file`: JSON file to load for defining experiments with multiple models/extractors/normalization steps. Overrides other parsed arguments.
- `-se`, `--slide_evaluation`: Enable slide evaluation mode. (Default: False)

When using the multi_input.json file multiple feature extractors and models can be used simultaneously, as follows:
```
{
   "normalization":[
      "macenko"
   ],
   "normalization_presets":[
      "v3"
   ],
   "feature_extractor":[
      "CTransPath",
      "resnet50_imagenet"
   ],
   "mil_model":[
      "CLAM_SB",
      "CLAM_MB",
      "Att_MIL"
   ],
}    
```
