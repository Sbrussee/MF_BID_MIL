## MF-CLAM
This is the repository for the CLAM model applied to the task of differentating early between Mycosis Fungoides (MF) and Benign Inflammatory Dermatoses (BIDs). The implementation heavily relies on the SlideFlow Python Package (ver 2.3.0).
Pleas note that this model **is not intended to be used for clinical practice**, instead it should be used for **research purposes only**.

## Publication
This repository contains the software behind this [publication](https://www.sciencedirect.com/science/article/pii/S0022202X24021018) in the Journal of Investigative Dermatology. When using the code or weights from this publication, please cite the following
```
@article{doeleman2024deep,
  title={Deep learning-based classification of early-stage mycosis fungoides and benign inflammatory dermatoses on hematoxylin and eosin-stained whole-slide images: a retrospective, proof-of-concept study},
  author={Doeleman, Thom and Brussee, Siemen and Hondelink, Liesbeth M and Westerbeek, Dani{\"e}lle WF and Sequeira, Ana M and Valkema, Pieter A and Jansen, Patty M and He, Junling and Vermeer, Maarten H and Quint, Koen D and others},
  journal={Journal of Investigative Dermatology},
  year={2024},
  publisher={Elsevier}
}
```

## Installation
In order to run the script, one can use pip to install the requirements:
```
pip install -r requirements.txt
```

## Using the model weights:

This guide will help you load the CLAM model using the `.pth` weights and `mil_params.json` configuration file, with the additional functionality of `CLAMModelConfig` from `slideflow-gpl`.

## Requirements

Make sure you have the following installed:

- Python 3.7+
- [SlideFlow](slideflow.dev) installed via `pip install slideflow`
- Optionally, for loading the CLAM model, we install slideflow-gpl via `pip install slideflow-gpl`
- A valid Whole Slide Image (WSI) or dataset with extracted tile features

## Files Provided

- `model_weights.pth`: The pre-trained weights of the CLAM model
- `mil_params.json`: The configuration file for building the model

Ensure these two files are located in the same directory when performing inference or evaluation. Users can simply load the weights onto a CLAM model with the right configuration, or use slideflow's prediction and evaluation capabilities to use the model for their own slides. You can load the CLAM model configuration from mil_params.json using the CLAMModelConfig class from slideflow-gpl.
```python
import slideflow as sf
from slideflow.clam import CLAMModelConfig

# Define the paths
model_path = '/path_to_model_directory'  # Directory containing model_weights.pth and mil_params.json
config_path = '/path_to_model_directory/mil_params.json'

# Step 1: Load the JSON configuration
with open(config_path, 'r') as f:
    config_data = json.load(f)

# Instantiate CLAMModelConfig using the loaded JSON parameters
clam_config = CLAMModelConfig(
    model=config_data.get("model", "clam_sb"),
    model_size=config_data["params"].get("model_size", "small"),
    bag_loss=config_data["params"].get("bag_loss", "ce"),
    bag_weight=config_data["params"].get("bag_weight", 0.7),
    dropout=config_data["params"].get("dropout", False),
    opt=config_data["params"].get("opt", "adam"),
    inst_loss=config_data["params"].get("inst_loss", "ce"),
    no_inst_cluster=config_data["params"].get("no_inst_cluster", False),
    B=config_data["params"].get("B", 8)
)

# Build the model
n_in = 1024    # Number of input features (feature size) UNI: 1024
n_out = 2      # Number of output classes (e.g., binary classification)

# This builds the CLAM model with the specified input and output dimensions
clam_model = config.build_model(n_in=n_in, n_out=n_out)

# Load the pre-trained weights (assuming the weights file is in the same directory)
weights_path = f'{model_path}/best_mf_clam_weights.pth'

# Optionally, provide a feature extractor and normalizer, or let SlideFlow auto-detect them
extractor = None
normalizer = None

# Generate predictions for a slide
slide_path = '/path_to_slide'
predictions, attention_scores = sf.mil.predict_slide(
    model=model_path,
    slide=slide_path,
    extractor=extractor,
    normalizer=normalizer,
    attention=True  # Set to True if you want attention scores
)

# Print predictions and attention scores
print("Predictions:", predictions)
print("Attention Scores:", attention_scores)
```
Parameters for CLAMModelConfig:
- model: Specify the model architecture. Options: 'clam_sb', 'clam_mb', 'mil_fc', 'mil_fc_mc'.
- model_size: Available sizes include small, big, multiscale, xception, etc.
- bag_loss: The bag loss function, either 'ce' (Cross-Entropy) or 'svm' (SmoothTop1SVM).
- bag_weight: Weight for the bag loss, usually between 0 and 1.
- dropout: Whether to apply dropout (True or False).
- opt: The optimizer, either 'adam' or 'sgd'.
- inst_loss: Instance loss function, either 'ce' or 'svm'.
- no_inst_cluster: Disable instance-level clustering (True or False).
- B: Number of positive/negative patches to sample for instance-level training.
- validate: Whether to validate the hyperparameter configuration.
The configuration file mil_params.json will automatically load these settings, and you can customize them as needed.

3. Evaluating the CLAM Model
To evaluate the CLAM model on a dataset, use the eval_mil function, similar to how you would evaluate any MIL model in SlideFlow.

```python
import slideflow as sf
from slideflow.clam import CLAMModelConfig

# Define the paths and dataset
model_path = '/path_to_model_directory/'
weights_path = '/path_to_model_directory/model_weights.pth'
config_path = '/path_to_model_directory/mil_params.json'
dataset = sf.Dataset('/path_to_dataset')  # Path to your dataset
outcomes = "category"  # The label(s) or category of outcomes
bags = '/path_to_bags_directory'  # Directory containing bag files

# Evaluate the model
results = sf.mil.eval_mil(
    weights=model_path,
    dataset=dataset,
    outcomes=outcomes,
    bags=bags,
    attention_heatmaps=True  # Set to True if you want attention heatmaps
)
```
Results will be saved to the default 'mil' directory, or specify with 'outdir'
### Directory Structure
Make sure the .pth weights file and mil_params.json configuration file are located in the same directory as shown below:
```
bash
/path_to_model_directory/
├── model_weights.pth
├── mil_params.json
```
4. Additional Features
You can customize the following options when calling predict_slide or eval_mil:

- Feature Extractor: Specify a custom feature extractor if needed.
- Stain Normalizer: Provide a custom stain normalizer for different stain normalization techniques.
- Attention Pooling: Control attention pooling strategies for attention scores (average or max).
- Attention Heatmaps: Generate attention heatmaps for slides by enabling the attention_heatmaps flag.

For more information, check the [SlideFlow documentation](slideflow.dev) documentation.

## Troubleshooting
- Ensure that mil_params.json and .pth weights are in the same directory.
- Ensure your dataset and bags are structured correctly.
- If any errors occur, check the console logs for additional details and adjust paths accordingly.

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
