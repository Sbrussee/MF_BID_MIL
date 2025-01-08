This repository contains the inference code for the publication "Deep Learning–Based Classification of Early-Stage Mycosis Fungoides and Benign Inflammatory Dermatoses on H&E-Stained Whole-Slide Images: A Retrospective, Proof-of-Concept Study", available at https://www.sciencedirect.com/science/article/pii/S0022202X24021018?via%3Dihub 

The weights are part of the repository. Below is the code, as present in get_predictions.py, which can be used to perform inference using the trained weights. 

The code in this repository is for RESEARCH PURPOSES ONLY, and should not be used in clinical practice.

# Required imports
```
import slideflow as sf
from slideflow.mil import ModelConfigCLAM, mil_config
from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
from slideflow.model.extractors import register_torch
from huggingface_hub import login, hf_hub_download

import json
import timm
import os
import torch
from torchvision import transforms
```
# Define the paths
```
model_path = './'  # Directory containing model_weights.pth and mil_params.json
config_path = './mil_params.json'
```


# Step 1: Load the JSON configuration
```
with open(config_path, 'r') as f:
    config_data = json.load(f)
```
# Instantiate model config using the loaded JSON parameters
```
clam_config = mil_config(
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
```

# Build the model
```
n_in = 1024    # Number of input features (feature size) UNI: 1024
n_out = 2      # Number of output classes (e.g., binary classification)

# This builds the CLAM model with the specified input and output dimensions
clam_model = clam_config.build_model()

# Load the pre-trained weights (assuming the weights file is in the same directory)
weights_path = f'best_mf_clam_weights.pth'

if config_data['weights'] != weights_path:
    config_data.update({"weights" : weights_path})
    with open(config_path, 'w') as file:
        json.dump(config_data, file)
#Instantiate UNI model (requires huggingface token)
login(token="your_huggingface_token")

@register_torch
class uni(TorchFeatureExtractor):
    """
    UNI feature extractor, with ViT-Large backbone

    Parameters
    ----------
    tile_px : int
        The size of the tile
    
    Attributes
    ----------
    model : VisionTransformer
        The Vision Transformer model
    transform : torchvision.transforms.Compose
        The transformation pipeline
    preprocess_kwargs : dict
        The preprocessing arguments
    

    Methods
    -------
    dump_config()
        Dump the configuration of the feature extractor
    """
    tag = "uni"

    def __init__(self, tile_px=256):
        super().__init__()
        local_dir = "weights_directory"
        model_name = "uni.bin"
        model_temp_name = "pytorch_model.bin"
        model_path = os.path.join(local_dir, model_name)

        if not os.path.exists(model_path):
            temp_model_path = hf_hub_download(repo_id="MahmoodLab/UNI", filename=model_temp_name, local_dir=local_dir, force_download=True)
            os.rename(temp_model_path, model_path)
        
        self.model = timm.create_model(
           "vit_large_patch16_224", img_size=224,  init_values=1e-5, num_classes=0
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        self.model.to('cuda')
        self.num_features = 1024 
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'uni',
            'kwargs': {}
        }
```
# Optionally, provide a feature extractor and normalizer, or let SlideFlow auto-detect them
```
extractor = sf.model.build_feature_extractor('uni', tile_px=256) # Assuming you use tile sizes of 256, UNI will resize them to 224.
normalizer = sf.norm.StainNormalizer(method=config_data["bags_extractor"]["normalizer"]["method"]) # Macenko stain normalization
```
# Generate predictions for a single slide
```
slide_path = '/path/to/slide.tiff'
predictions, attention_scores = sf.mil.predict_slide(
    model=model_path,
    slide=slide_path,
    extractor=extractor,
    normalizer=normalizer,
    attention=True  # Set to True if you want attention scores
)

# Print predictions and attention scores
print("Predictions:", predictions)
# Get the predicted label
predicted_label = predictions.argmax()
print("Predicted label:", predicted_label)
print("Attention Scores:", attention_scores)

# Save the predictions and attention scores
output_path = './predictions.json'
with open(output_path, 'w') as f:
    json.dump({"predictions": predictions.tolist(), "attention_scores": attention_scores.tolist()}, f)

print(f"Predictions and attention scores saved to {output_path}")
```
# Multiple slide predictions
```
slide_directory = "path/to/slides"
#Get list of slides
slides = os.listdir(slide_directory)
#Get predicted labels and probabilities for each slide
probabilities = []
labels = []
for slide in slides:
    slide_path = os.path.join(slide_directory, slide)
    predictions, _ = sf.mil.predict_slide(
        model=model_path,
        slide=slide_path,
        extractor=extractor,
        normalizer=normalizer,
        attention=False
    )
    probabilities.append(predictions)
    labels.append(predictions.argmax())
#Save table of slide name, predicted label and probabilities
output_path = './slide_predictions.csv'
with open(output_path, 'w') as f:
    f.write("Slide,Predicted Label,Probability\n")
    for slide, label, prob in zip(slides, labels, probabilities):
        f.write(f"{slide},{label},{prob}\n")
print(f"Slide predictions saved to {output_path}")

````
