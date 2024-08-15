import os
import numpy as np
from PIL import Image


def get_top5_annotation_tiles_per_class(attention_map, bag_index, tile_directory, slide_name, heatmap, diagnosis):
    #Get the top 5 attention values
    top5_attention_values = np.argsort(attention_map)[::-1][:5]
    print(f"Top 5 attention indices: {top5_attention_values}")
    top5_least_attended = np.argsort(attention_map)[:5]

    #Get the corresponding tile coordinates from bag_index
    top5_tile_coordinates = bag_index[top5_attention_values]
    print(f"Top 5 tile coordinates: {top5_tile_coordinates}")
    top5_least_attended_coordinates = bag_index[top5_least_attended]

    #Get the corresponding tile paths
    top5_tiles = [f"{tile_directory}/{slide_name}-{coord[0]}-{coord[1]}.png" for coord in top5_tile_coordinates]
    top5_least_attended_tiles = [f"{tile_directory}/{slide_name}-{coord[0]}-{coord[1]}.png" for coord in top5_least_attended_coordinates]
    print(f"Top 5 tile paths: {top5_tiles}")

    #Construct top 5 tiles directory
    os.makedirs(f"top5_tiles", exist_ok=True)

    #Plot the top 5 tiles using matplotlib
    import matplotlib.pyplot as plt
    import cv2
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i, tile in enumerate(top5_tiles):
        img = Image.open(tile)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        axs[i].imshow(img)
        axs[i].axis('off')
    #Set global title above the subplots
    plt.suptitle(f"Top 5 most attended tiles ({diagnosis} {slide_name})", fontsize=21)
    plt.tight_layout()
    plt.savefig(f"top5_tiles/{diagnosis}_{slide_name}_top5_tiles.png")
    plt.close()

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i, tile in enumerate(top5_least_attended_tiles):
        #Read image using PIL
        tile = Image.open(tile)
        if tile.mode != 'RGB':
            tile = tile.convert('RGB')
        axs[i].imshow(tile)
        axs[i].axis('off')
    plt.suptitle(f"Top 5 least attended tiles ({diagnosis} {slide_name})", fontsize=21)
    plt.tight_layout()
    plt.savefig(f"top5_tiles/{diagnosis}_{slide_name}_least_attended_tiles.png")
    plt.close()




def main():
    os.system(f"rm top5_tiles/*")

    list_of_MF_slides = ["H22-01898-1A", "H21-09823-1A", "R23-83334-2A", "H23-12427-1A", "R23-80063-1A"]
    list_of_BID_slides = ["H22-02358-1A", "R21-81541-2A", "R21-81541-1A", "R22-82630-1A", "R22-82630-2A" ]
    for MF_slide_to_evaluate, BID_slide_to_evaluate in zip(list_of_MF_slides, list_of_BID_slides):
        directory_with_results = "/exports/path-cutane-lymfomen-hpc/siemen/PathBench/mf_clam/mil_eval/00133_clam_sb_uni_macenko_ext_set"
        
        MF_attention = np.load(f"{directory_with_results}/00000-clam_sb/attention/{MF_slide_to_evaluate}_att.npz")['arr_0']
        BID_attention = np.load(f"{directory_with_results}/00000-clam_sb/attention/{BID_slide_to_evaluate}_att.npz")['arr_0']
        print(MF_attention.shape)
        print(BID_attention.shape)
        MF_tile_directory = f"/exports/path-cutane-lymfomen-hpc/siemen/PathBench/mf_clam/tiles/ext_set/256px_20x/{MF_slide_to_evaluate}"
        BID_tile_directory = f"/exports/path-cutane-lymfomen-hpc/siemen/PathBench/mf_clam/tiles/ext_set/256px_20x/{BID_slide_to_evaluate}"
        MF_bag_directory = np.load(f"/exports/path-cutane-lymfomen-hpc/siemen/PathBench/mf_clam/bags/uni_macenko_20x_256_ext_set/{MF_slide_to_evaluate}.index.npz")['arr_0']
        BID_bag_directory = np.load(f"/exports/path-cutane-lymfomen-hpc/siemen/PathBench/mf_clam/bags/uni_macenko_20x_256_ext_set/{BID_slide_to_evaluate}.index.npz")['arr_0']
        print(MF_bag_directory.shape, BID_bag_directory.shape)
        MF_heatmap = f"{directory_with_results}/00000-clam_sb/heatmaps/{MF_slide_to_evaluate}_attn.png"
        BID_heatmap = f"{directory_with_results}/00000-clam_sb/heatmaps/{BID_slide_to_evaluate}_attn.png"

        #Print the number of attention values, and number of tiles
        print(f"MF_attention: {MF_attention.shape}")
        print(f"BID_attention: {BID_attention.shape}")
        print(f"MF_tile_directory: {len(os.listdir(MF_tile_directory))}")
        print(f"BID_tile_directory: {len(os.listdir(BID_tile_directory))}")

        get_top5_annotation_tiles_per_class(MF_attention, MF_bag_directory, MF_tile_directory, MF_slide_to_evaluate, MF_heatmap, "MF")
        get_top5_annotation_tiles_per_class(BID_attention, BID_bag_directory, BID_tile_directory, BID_slide_to_evaluate, BID_heatmap, "BID")


    














if __name__ == "__main__":
    main()