import slideflow as sf
from slideflow.mil import mil_config
import slideflow.mil as mil
from slideflow.stats.metrics import ClassifierMetrics
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision import transforms
from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
from slideflow.model.extractors import register_torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import pandas as pd
import os
import re
import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import neptune

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models.vision_transformer import VisionTransformer
from huggingface_hub import login, hf_hub_download

os.environ['NEPTUNE_API_TOKEN'] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MWI4Zjg5My04NTdlLTQxMGItYTcwZC02ZGIwMmE0NWUwMjQifQ=="
os.environ['NEPTUNE_WORKSPACE'] = "siemenbrussee"
os.environ['NETPUNE_PROJECT'] = "siemenbrussee/MF-CLAM"

run = neptune.init_run(
    project="siemenbrussee/MF-CLAM",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MWI4Zjg5My04NTdlLTQxMGItYTcwZC02ZGIwMmE0NWUwMjQifQ==",
)

parser = argparse.ArgumentParser()
#Global arguments
parser.add_argument('-p', '--project_directory',
                    help="Directory to store the project")
parser.add_argument('-s', '--slide_directory',
                    help="Directory where slides are located")
parser.add_argument('-a', '--annotation_file',
                    help="CSV file having the slide id's, labels and patient id's. It should, at least, contain a'slide' and 'patient' column.")
parser.add_argument('-tf', '--test_fraction', type=float, default=0.1,
                    help="Fraction of dataset to hold apart for testing.")
parser.add_argument('-k', '--k_fold', type=int, default=10,
                    help="number of folds to use for k-fold cross-validation")
parser.add_argument('-ts', '--tile_size', type=int, default=256,
                    help="Size of tiles to use in pixels")
parser.add_argument('-mg', '--magnification', choices=['40x', '20x', '10x', '5x'], default='40x',
                    help="Magnification level to use")
parser.add_argument('-ag', '--augmentation', type=str, default='xyjrbn',
                    help="augmentation methods to use. Can be any combination of x: random x-flip, y: random y-flip, r: random cardinal rotation,\
                     j: random JPEG compression, b: random gaussian blur, n: stain augmentation. e.g. 'xyjn'")
parser.add_argument('-l', '--aggregation_level', choices=['patient', 'slide'],
                    help="Level of bag aggregation to use, can be 'patient' or 'slide'.")
parser.add_argument('-b', '--training_balance', choices=['tile', 'slide', 'patient', 'category'],
                    help="Balances batches for training. tile causes each tile to be sampled with equal probability,\
                    slide causes batches to be sampled on the same slide with equal probability,\
                    patient causes batches to be sampled from the same patient with equal probability,\
                    category makes sure the categorical outcome labeled are sampled with equal probability")

#Set feature extractor
parser.add_argument('-f', '--feature_extractor', choices=['CTransPath', 'uni', 'dino', 'RetCCL', 'HistoSSL', 'PLIP', 'SimCLR', 'DinoV2', 'resnet50_imagenet', 'barlow_twins_feature_extractor'],
                    help="Pretrained feature extractors to use", default="RetCCL")

#Set MIL model
parser.add_argument('-m', '--model', choices=['Attention_MIL', 'CLAM_SB', 'CLAM_MB', 'MIL_fc', 'MIL_fc_mc', 'TransMIL'],
                    help="MIL model to use", default="Attention MIL")

#Set normalization parameters
parser.add_argument('-n', '--normalization', choices=['macenko', 'vahadane_sklearn', 'reinhard', 'cyclegan', 'None'], default="macenko",
                    help="Normalization method to use, the parameter preset is set using --stain_norm_preset ")

parser.add_argument('-sp', '--stain_norm_preset', choices=['v1', 'v2', 'v3'], default='v3',
                    help="Stain normalization preset parameter sets to use.")

parser.add_argument('-j', '--json_file', default=None,
                    help="JSON file to load for defining experiments with multiple models/extractors/normalization steps. Overrides other parsed arguments.")

parser.add_argument('-se', '--slide_evaluation', default=False, action='store_true')
args = parser.parse_args()
#Print chosen arguments
print(args)
#Print available feature extractors
print("Available feature extractors: ", sf.model.list_extractors())

# Check if GPU is available
if torch.cuda.is_available():
    # Get the current GPU device
    device = torch.cuda.current_device()

    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(device)

    print(f'Using GPU: {gpu_name}')
else:
    print('GPU not available. Using CPU.')

#Check if JSON file should be used
if args.json_file != None:
    with open(args.json_file, "r") as params_file:
        params = json.load(params_file)

login(token="")

if args.feature_extractor == 'uni':
    local_dir = "./pretrained_weights"
    model_name = "pytorch_model.bin"
    hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir)


# Define the ResNet model with a custom trunk
class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# Define the URL for downloading the pretrained weights from lunit
def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

# Download the pretrained weights from lunit
def download_pretrained_weights(key, destination):
    pretrained_url = get_pretrained_url(key)
    if not os.path.exists(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        print(f"Downloading pretrained weights for {key}...")
        torch.hub.download_url_to_file(pretrained_url, destination)
        print("Pretrained weights downloaded successfully.")
    else:
        print(f"Pretrained weights for {key} already exist, skipping download.")

def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=False)
        )
        print(verbose)
    return model

def resnet50(pretrained, progress, key, **kwargs):
    """
    Arguments:
        pretrained: bool
        progress: bool
        key: str

    This function loads the ResNet50 model with the pretrained weights from lunit.

    Returns:
        model: ResNetTrunk
    """
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weights_path = f"pretrained_weights/{key}.torch"
        pretrained_url = get_pretrained_url(key)
        download_pretrained_weights(key, weights_path)
        model.load_state_dict(torch.load(weights_path))
    return model

@register_torch
class dino(TorchFeatureExtractor):
    
        tag = 'dino'
    
        def __init__(self, tile_px=256):
            super().__init__()
    
            # Load DINO pre-trained weights
            self.model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
            self.model.to('cuda')
            self.model.eval()

            self.num_features = 384  # Assuming Vision Transformer output features of size 384

            # Image preprocessing.
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.Lambda(lambda x: x / 255.),
                #Convert to float tensor
                transforms.ConvertImageDtype(torch.float32),
                #Transform to float tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            # Disable Slideflow standardization,
            # as we are standardizing with transforms.Normalize
            self.preprocess_kwargs = {'standardize': False}

            def dump_config(self):
                return {
                    'class': 'dino',
                    'kwargs': {}
                }

# Define the feature extractor for Barlow Twins
@register_torch
class barlow_twins_feature_extractor(TorchFeatureExtractor):

    tag = 'barlow_twins_feature_extractor'

    def __init__(self, tile_px=256):
        super().__init__()

        # Load ResNet50 trunk with Barlow Twins pre-trained weights
        self.model = resnet50(pretrained=True, progress=False, key="BT")
        self.model.to('cuda')  # Move model to GPU if available
        self.model.eval()  # Set the model to evaluation mode

        # Set the number of features generated by the model
        self.num_features = 2048  # Assuming ResNet50 output features of size 2048

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.)
        ])

        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': True}

    def dump_config(self):
        return {
            'class': 'barlow_twins_feature_extractor',
            'kwargs': {}
        }

@register_torch
class uni(TorchFeatureExtractor):
    tag="uni"

    def __init__(self, tile_px=256):
        super().__init__()

        self.model = timm.create_model(
       "vit_large_patch16_224", img_size=224,  init_values=1e-5, num_classes=0
        )
        
        self.model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        self.model.to('cuda')
        self.num_features = 1024 
        self.transform = transforms.Compose(
         [
        transforms.Resize(224),
        #Transform to float tensor
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ]
        )
        self.model.eval()
        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {
            'class': 'uni',
            'kwargs': {}
        }

# Define the feature extractor for MoCoV2
@register_torch
class mocov2_feature_extractor(TorchFeatureExtractor):

    tag = 'mocov2_feature_extractor'

    def __init__(self, tile_px=256):
        super().__init__()

        # Load ResNet50 trunk with Barlow Twins pre-trained weights
        self.model = resnet50(pretrained=True, progress=False, key="MoCoV2")
        self.model.to('cuda')  # Move model to GPU if available
        self.model.eval()  # Set the model to evaluation mode

        # Set the number of features generated by the model
        self.num_features = 2048  # Assuming ResNet50 output features of size 2048

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.)
        ])

        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': True}

    def dump_config(self):
        return {
            'class': 'mocov2_feature_extractor',
            'kwargs': {}
        }


# Define the feature extractor for SwAV
@register_torch
class swav_feature_extractor(TorchFeatureExtractor):
    tag = 'swav_feature_extractor'

    def __init__(self, tile_px=256):
        super().__init__()

        # Load ResNet50 trunk with Barlow Twins pre-trained weights
        self.model = resnet50(pretrained=True, progress=False, key="SwAV")
        self.model.to('cuda')  # Move model to GPU if available
        self.model.eval()  # Set the model to evaluation mode

        # Set the number of features generated by the model
        self.num_features = 2048  # Assuming ResNet50 output features of size 2048

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.)
        ])

        # Slideflow standardization
        self.preprocess_kwargs = {'standardize': True}

    def dump_config(self):
        return {
            'class': 'swav_feature_extractor',
            'kwargs': {}
        }


def process_annotation_file(original_path):
    """
    Arguments:
        original_path: str

    This function processes the annotation file by removing duplicate rows and saving the processed file as a new CSV file.

    Returns:
        None
    """
    df = pd.read_csv(original_path, header=0, sep=";")
    df.rename(columns={'case_id' : 'patient', 'slide_id' : 'slide'}, inplace=True)
    # Finding duplicate rows based on 'slide' column
    duplicate_rows = df[df.duplicated(subset='slide', keep=False)]

    # Printing duplicate rows
    print("Duplicate Rows based on 'slide' column:")
    print(duplicate_rows)
    #df['slide'] = df['slide'].apply(lambda x:x + '.tiff')
    df.drop_duplicates(inplace=True, subset="slide")
    #rows_to_drop = df[(df['patient'].isin([3787608, 8839385, 9476237, 1093978])) & (df['category'] == 'BID')].index
    #df = df.drop(rows_to_drop)
    print("Processed annotation file: ", df)
    df.to_csv(f"{os.path.basename(original_path).strip('.csv')}_slideflow.csv", index=False,sep=",")

def get_highest_numbered_filename(directory_path):
    """
    Arguments:
        directory_path: str

    This function returns the highest numbered filename in the specified directory.

    Returns:
        highest_number_part: str
    """
    # List all files in the directory
    files = os.listdir(directory_path)

    # Initialize variables to keep track of the highest number and corresponding filename
    highest_number = float('-inf')
    highest_number_filename = None

    # Iterate over each file
    for filename in files:
        # Get the part before the first '-'
        first_part = filename.split('-')[0]

        # Try to convert the first part to a number
        try:
            number = int(first_part)
            # If the converted number is higher than the current highest, update the variables
            if number > highest_number:
                highest_number = number
                highest_number_part = first_part
        except ValueError:
            pass  # Ignore non-numeric parts

    print(highest_number_part)
    return highest_number_part


def plot_roc_curve_across_splits(tpr_list, fpr_list, model, extractor, normalizer, dataset_name, tile_size, magnification):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for i in range(len(fpr_list)):
        fpr = fpr_list[i]
        tpr = np.interp(mean_fpr, fpr, tpr_list[i])
        tpr[0] = 0.0
        tprs.append(tpr)
        roc_auc = auc(fpr_list[i], tpr_list[i])
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot the ROC curve with shaded standard deviation area
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(
        f"{args.project_directory}/mil_eval/{model.lower()}_{extractor.lower()}_{normalizer.lower()}_roc_auc_test_{dataset_name}_{tile_size}_{magnification}.png"
    )

# Calculate the misclassification weights
def calculate_weights(df):
    """
    Arguments:
        df: pd.DataFrame

    This function calculates weights based on the occurence of each class in the dataset.

    Returns:
        weights: dict
    """
    counts = df['category'].value_counts()
    weight_bids = 1 / counts.get('BID', 1)
    weight_mfs = 2 / counts.get('MF', 1)  # 2x the weight for MFs
    return {'BID': weight_bids, 'MF': weight_mfs}

def split_dataset_by_patient(dataset, test_fraction=0.2):
    """
    Arguments:
        dataset: sf.Dataset
        test_fraction: float

    This function splits the dataset into training and validation sets based on the patient ID.

    Returns:
        train: sf.Dataset
        test: sf.Dataset
    """
    df = pd.read_csv(args.annotation_file)
    df = df[df['dataset'] == 'train']

    slides_by_patient = df.groupby("patient")['slide'].apply(list).reset_index()
    slides_by_patient = slides_by_patient.sample(frac=1).reset_index(drop=True)
    print(slides_by_patient)

    train_patients, val_patients = train_test_split(slides_by_patient, test_size=test_fraction, random_state=42)

    # Extract slides for training and validation sets
    train_slides = df[df['patient'].isin(train_patients['patient'])]
    val_slides = df[df['patient'].isin(val_patients['patient'])]

    # Save training and validation sets to new CSV files
    train_slides.to_csv('train_slides.csv', index=False)
    val_slides.to_csv('val_slides.csv', index=False)

   # Assert no leakage of patients between training and validation sets
    assert set(train_patients['patient']).intersection(set(val_patients['patient'])) == set(), "Patient leakage detected!"

    print("No patient leakage!.")

    train, test = sf.Dataset(
    slides=args.slide_directory,
    tfrecords=f"{args.project_directory}/tfrecords",
    annotations='train_slides.csv',
    tile_px = args.tile_size,
    tile_um = args.magnification
    ), sf.Dataset(
        slides=args.slide_directory,
        tfrecords=f"{args.project_directory}/tfrecords",
        annotations='val_slides.csv',
        tile_px = args.tile_size,
        tile_um = args.magnification
    )

    print(train, test)
    train = train.balance(headers='category', strategy=args.training_balance)
    return train, test

def read_validation_set(project):
    """
    Arguments:
        project: sf.Project

    This function reads the external validation set and adds it to the project.

    Returns:
        project: sf.Project
    """
    #process_annotation_file("../../Thom_Doeleman/annotations.csv")

    project.add_source(
    name="ext_set",
    slides="../../Thom_Doeleman/CLAM_validate_cropped",
    roi="../../Thom_Doeleman/CLAM_validate_cropped/rois",
    tfrecords=f"{args.project_directory}/tfrecords/ext_set",
    tiles=f"{args.project_directory}/tiles/ext_set"
    )
    test_set = project.dataset(
    sources=['ext_set'],
    filters = {'dataset' : 'validate'},
    tile_px=args.tile_size,
    tile_um=args.magnification
    )

    project.extract_tiles(
            qc='both', #Both use Otsu Thresholding and Blur detection
            qc_filter_threshold=0.8,
            source='ext_set',
            tile_px=args.tile_size,
            tile_um=args.magnification,
            save_tiles=True,
            img_format='png',
            enable_downsample=False
            )

    return project, test_set


def train_mil_model(train, val, test, model, extractor, normalizer, project, config):
    """
    Arguments:
        train: sf.Dataset
        val: sf.Dataset
        test: sf.Dataset
        model: str
        extractor: str
        normalizer: str
        project: sf.Project
        config: dict

    This function trains a MIL model using the specified configuration.

    Returns:
        result_frame: pd.DataFrame
    """

    config = mil_config(model = model.lower(),
    aggregation_level = args.aggregation_level,
    trainer='fastai',
    epochs=32,
    batch_size=32)

    print(config)
    if args.aggregation_level == 'patient':
        project.train_mil(
        config=config,
        outcomes="category",
        train_dataset=train,
        val_dataset=val,
        bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_{args.magnification}_{args.tile_size}",
        #attention_heatmaps=True,
        #cmap="coolwarm",
        exp_label=f"{model.lower()}_{extractor.lower()}_{normalizer.lower()}"
        )

        current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}/mil/")

        if args.slide_evaluation == True:
            config =  mil_config(args.model.lower(),
            aggregation_level='slide',
            epochs=32)

            result_frame = mil.eval_mil(
            weights=f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
            outcomes="category",
            dataset=test,
            bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_{args.magnification}_{args.tile_size}",
            config=config,
            outdir=f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
            attention_heatmaps=True,
            cmap="jet",
            norm="linear"
        )
        else:
            result_frame = mil.eval_mil(
            weights=f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
            outcomes="category",
            dataset=test,
            bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_{args.magnification}_{args.tile_size}",
            config=config,
            outdir=f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
            #attention_heatmaps=True,
            #cmap="coolwarm"
        )
    elif args.aggregation_level == 'slide':
        project.train_mil(
        config=config,
        outcomes="category",
        train_dataset=train,
        val_dataset=val,
        bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_{args.magnification}_{args.tile_size}",
        attention_heatmaps=True,
        cmap="jet",
        norm="linear",
        exp_label=f"{model.lower()}_{extractor.lower()}_{normalizer.lower()}"
        )

        current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}/mil/")

        result_frame = mil.eval_mil(
        weights=f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
        outcomes="category",
        dataset=test,
        bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_{args.magnification}_{args.tile_size}",
        config=config,
        outdir=f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
        attention_heatmaps=True,
        cmap="jet",
        norm="linear"
        )
    return result_frame


def visualize_results(result_frame, model, extractor, normalizer, ext_set=False):
    """
    Arguments:
        result_frame: pd.DataFrame
        model: str
        extractor: str
        normalizer: str
        ext_set: bool

    This function visualizes the results of the MIL model.

    Returns:
        result_frame: pd.DataFrame
        balanced_accuracy: float
        auroc: float
    """
    current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}/mil/")
    y_pred_cols = [c for c in result_frame.columns if c.startswith('y_pred')]
    for idx in range(len(y_pred_cols)):
        m = ClassifierMetrics(
            y_true=(result_frame.y_true.values == idx).astype(int),
            y_pred=result_frame[f'y_pred{idx}'].values
        )

        fpr, tpr, auroc, threshold = m.fpr, m.tpr, m.auroc, m.threshold
        optimal_idx = np.argmax(tpr-fpr)
        optimal_threshold = threshold[optimal_idx]
        y_pred_binary = (result_frame[f'y_pred{idx}'].values > optimal_threshold).astype(int)
        conf_mat = confusion_matrix(y_true=(result_frame.y_true.values == idx).astype(int), y_pred=y_pred_binary, normalize='all', labels=np.array([0,1]))
        display = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['MF', 'BID'])
        display.plot()
        if ext_set:
            plt.savefig(f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}_ext_set/conf_mat_test.png")
        else:
            plt.savefig(f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}/conf_mat_test.png")
        balanced_accuracy = balanced_accuracy_score((result_frame.y_true.values == idx).astype(int), y_pred_binary)
        print(f"BA cat #{idx}: {balanced_accuracy}")


    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="orange",
        lw=lw,
        label=f"ROC curve (area = %0.2f)" % auroc,
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC")
    plt.legend(loc="lower right")
    if ext_set:
        plt.savefig(f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}_ext_set/roc_auc_test.png")
    else:
        plt.savefig(f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}/roc_auc_test.png")
    #result_frame = pd.read_parquet(f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}/predictions.parquet", engine='pyarrow')

    return result_frame, balanced_accuracy, auroc, fpr, tpr

def main(validation=False):

    #Create project directory
    if not os.path.exists(args.project_directory):
        project = sf.create_project(
        root = args.project_directory,
        annotations = args.annotation_file,
        slides = args.slide_directory,
        )

    else:
        project = sf.load_project(args.project_directory, use_neptune=True)
        print("Neptune:", project.neptune_api, project.neptune_workspace )

    #Process annotation file
    if args.json_file != None:
        normalizers = params['normalization']
        extractors = params['feature_extractor']
        models = params['mil_model']

    else:
        normalizers = [args.normalization]
        extractors = [args.feature_extractor]
        models = [args.model]

    print(args.tile_size, args.magnification)
    #Create dataset
    dataset = project.dataset(tile_px=args.tile_size, tile_um=args.magnification, filters={'dataset' : 'train'})
    print(dataset)

    print("Tiling...")
    dataset.extract_tiles(
        qc='both', #Both use Otsu Thresholding and Blur detection
        qc_filter_threshold=0.8,
        save_tiles=True,
        img_format='png',
        enable_downsample=False
        )
    # Balance the dataset based on the outcome label, to ensure equal representation of each class
    dataset.balance(headers='category', strategy=args.training_balance)
    print("Splitting...")

    #Split the dataset into training and testing sets
    split_file = "exports/path-cutane-lymfomen-hpc/siemen/PathBench/splits.json"
    if os.path.exists(split_file):
        train, test = dataset.split(labels='category',  val_fraction=args.test_fraction, splits=split_file)
    else:
        train, test = dataset.split(labels='category', val_fraction=args.test_fraction)

    # Read external test set
    if validation:
        project, ext_test = read_validation_set(project)

    results = {}

    columns = ['normalization', 'feature_extractor', 'mil_model', 'split', 'balanced_accuracy', 'auc']
    df = pd.DataFrame(columns=columns)
    #Train models based on the specified parameters
    if validation:
        ext_df = pd.DataFrame(columns=columns)
    
    # Loop over all combinations of feature extractors, normalizers, and MIL models
    for extractor in extractors:
        for normalizer in normalizers:
            if normalizer.lower() == 'none':
                normalizer = None
            for model in models:
                feature_extractor = sf.model.build_feature_extractor(extractor.lower(), tile_px=args.tile_size)
                bag_directory = project.generate_feature_bags(feature_extractor,
                                                              dataset,
                                                              outdir=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_{args.magnification}_{args.tile_size}",
                                                              normalizer=normalizer,
                                                              normalizer_source=args.stain_norm_preset,
                                                              progress=False)

                #Set model configuration
                config = mil_config(args.model.lower(),
                aggregation_level=args.aggregation_level,
                epochs=32)
                #Split using specified k-fold
                splits = train.kfold_split(
                k=args.k_fold,
                labels="category",
                )
                split_index = 0
                best_roc_auc = 0
                int_test_tpr, int_test_fpr = [], []
                ext_test_tpr, ext_test_fpr = [], []
                for train, val in splits:
                    #Train model
                    result_frame = train_mil_model(train, val, test, model, extractor, normalizer, project, config)

                    result_frame, balanced_accuracy, roc_auc, fpr, tpr  = visualize_results(result_frame, model, extractor, normalizer)
                    int_test_tpr.append(tpr)
                    int_test_fpr.append(fpr)
                    current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}/mil/")

                    results["_".join([extractor, normalizer, model, str(split_index)])] = balanced_accuracy
                    print("Balanced Accuracy: ", balanced_accuracy)
                    data = {
                    'normalization' : normalizer,
                    'feature_extractor' : extractor,
                    'mil_model' : model,
                    'split': split_index,
                    'balanced_accuracy' : balanced_accuracy,
                    'auc' : roc_auc
                    }
                    df = df.append(data, ignore_index=True)
                    print(df)
                    print("Validating...")

                    # Validate model on external set
                    if validation:
                        feature_extractor = sf.model.build_feature_extractor(extractor.lower(), tile_px=args.tile_size)
                        bag_directory = project.generate_feature_bags(feature_extractor,
                                                                      ext_test,
                                                                      outdir=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_{args.magnification}_{args.tile_size}_ext_set",
                                                                      normalizer=normalizer,
                                                                      normalizer_source=args.stain_norm_preset,
                                                                      progress=False)


                        if args.slide_evaluation:
                            config =  mil_config(args.model.lower(),
                            aggregation_level='slide',
                            epochs=32)

                        current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}/mil/")

                        if args.slide_evaluation:
                            result_frame = mil.eval_mil(
                            weights=f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
                            outcomes="category",
                            dataset=ext_test,
                            bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_{args.magnification}_{args.tile_size}_ext_set",
                            config=config,
                            outdir=f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}_ext_set",
                            attention_heatmaps=True,
                            cmap="jet",
                            norm="linear"
                            )
                        else:
                            result_frame = mil.eval_mil(
                            weights=f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
                            outcomes="category",
                            dataset=ext_test,
                            bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_{args.magnification}_{args.tile_size}_ext_set",
                            config=config,
                            outdir=f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}_ext_set",
                            #attention_heatmaps=True,
                            #cmap="coolwarm"
                            )   

                        result_frame, balanced_accuracy, roc_auc, fpr, tpr = visualize_results(result_frame, model, extractor, normalizer, ext_set=True)
                        ext_test_tpr.append(tpr)
                        ext_test_fpr.append(fpr)
                        data = {
                        'normalization' : normalizer,
                        'feature_extractor' : extractor,
                        'mil_model' : model,
                        'split': split_index,
                        'balanced_accuracy' : balanced_accuracy,
                        'auc' : roc_auc
                        }
                        ext_df = ext_df.append(data, ignore_index=True)
                        print(ext_df)

                    print("One loop done...")
                    split_index += 1

                plot_roc_curve_across_splits(int_test_tpr, int_test_fpr, model, extractor, normalizer, 'internal', args.tile_size, args.magnification)
                if validation:
                    plot_roc_curve_across_splits(ext_test_tpr, ext_test_fpr, model, extractor, normalizer, 'external', args.tile_size, args.magnification)
                


    #Summarize over splits
    grouped_df = df.groupby(['normalization', 'feature_extractor', 'mil_model'])
    if validation:
        ext_grouped_df = ext_df.groupby(['normalization', 'feature_extractor', 'mil_model'])

    result_df = grouped_df.agg({
    'normalization' : 'first',
    'feature_extractor' : 'first',
    'mil_model' : 'first',
    'balanced_accuracy' : ['mean', 'std'],
    'auc' : ['mean', 'std']
    })

    if validation:
        ext_result_df = ext_grouped_df.agg({
        'normalization' : 'first',
        'feature_extractor' : 'first',
        'mil_model' : 'first',
        'balanced_accuracy' : ['mean', 'std'],
        'auc' : ['mean', 'std']
        })


    date = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

    # Save results to CSV files
    df.to_csv(f"{args.project_directory}/results_{date}_full_{args.magnification}_{args.tile_size}.csv", index=False)
    result_df.to_csv(f"{args.project_directory}/results_{date}_{args.magnification}_{args.tile_size}.csv", index=False)

    if validation:
        ext_df.to_csv(f"{args.project_directory}/ext_set_results_{date}_{args.magnification}_{args.tile_size}_full.csv", index=False)
        ext_result_df.to_csv(f"{args.project_directory}/ext_set_results_{date}_{args.magnification}_{args.tile_size}.csv", index=False)

    with open("test_run.pkl", 'wb') as f:
        pickle.dump(results, f)


    run.stop()

if __name__ == "__main__":
    annotations = "../../Thom_Doeleman/annotations.csv"
    if not os.path.exists(f"{os.path.basename(annotations).strip('.csv')}_slideflow.csv"):
        process_annotation_file(annotations)
    main(validation=True)
