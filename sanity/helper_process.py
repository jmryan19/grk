import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from functools import reduce
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import seaborn as sns

LABEL_INDEX = -3
Y_HAT_INDEX = -2
IDX_INDEX = -1

def process_merge(paths, epoch, distance_metric = 'l2', normalization = 'none'):
    all_dfs = []
    for i, path in enumerate(paths):
        dat = torch.load(path + str(epoch), map_location=torch.device('cpu'), weights_only=True)
        labels = dat[: , LABEL_INDEX]
        y_hats = dat[: , Y_HAT_INDEX]
        idx = dat[: , IDX_INDEX]
        activations = dat[: , :-3]
        uniq_labels = np.unique(labels)
        
        global_mean = activations.mean(axis=0)
        ordered_l2 = []
        ordered_labels = []
        ordered_idx = []
        ordered_y_hats = []
        
        for j in uniq_labels:
            labels_indexing = (labels == j)
            class_activations = activations[labels_indexing, :]
            class_mean = class_activations.mean(axis=0) - global_mean

            if normalization == 'class':
                class_std = class_activations.std(axis=0) + 0.0001
                class_activations = class_activations / repeat(class_std, 'u -> n u', n = class_activations.shape[0])
                class_mean = class_mean / class_std
                # print(class_std)

            elif normalization == 'global':
                global_std = activations.std(axis=0) + 0.0001
                class_activations = class_activations - repeat(global_mean, 'u -> n u', n = class_activations.shape[0])
                class_activations = class_activations / repeat(global_std, 'u -> n u', n = class_activations.shape[0])
                class_mean = class_mean / global_std

            if distance_metric == 'cosine':
                cos_dist = 1 - torch.nn.functional.cosine_similarity(class_activations, repeat(class_mean, 'u -> n u', n = class_activations.shape[0]))
                _ = [ordered_l2.append(diff) for diff in cos_dist]
            
            elif distance_metric == 'l2':
                class_diff = class_activations - repeat(class_mean, 'u -> n u', n = class_activations.shape[0])
                _ = [ordered_l2.append(torch.linalg.vector_norm(diff)) for diff in class_diff]
            
            _ = [ordered_labels.append(lab) for lab in labels[labels_indexing]]
            _ = [ordered_idx.append(ids) for ids in idx[labels_indexing]]
            _ = [ordered_y_hats.append(y_hat) for y_hat in y_hats[labels_indexing]]
        
        all_together = torch.vstack([torch.tensor(x) for x in [ordered_l2, ordered_labels, ordered_idx, ordered_y_hats]]).T
        df = pd.DataFrame(all_together, columns = [f'dist_{i}', f'labels', f'idx', f'y_hats_{i}'])
        if i != 0:
            df = df.drop(columns=['labels'])
        
        all_dfs.append(df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='idx', how='outer'), all_dfs)
    return merged_df


def graph_hists(merged_df):
    l2_columns = [col for col in merged_df.columns if col.startswith('dist_')]
    num_cols = len(l2_columns)
    
    # Create a grid: 2 rows (general + label-wise), N columns (1 per l2_x)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), sharey='row')
    
    # Row 1: General histograms
    for i, col in enumerate(l2_columns):
        ax = axes[0, i]
        sns.histplot(merged_df[col], bins=30, kde=False, ax=ax, color='skyblue')
        ax.set_title(f"{col} - Overall")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
    
    # Row 2: Label-wise histograms
    for i, col in enumerate(l2_columns):
        ax = axes[1, i]
        for label in sorted(merged_df['labels'].unique()):
            subset = merged_df[merged_df['labels'] == label]
            sns.histplot(subset[col], bins=30, kde=False, label=f"Label {label}", alpha=0.5, ax=ax)
        ax.set_title(f"{col} - By Label")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Group by 'labels' and calculate mean and std
    grouped_stats = merged_df.groupby('labels')[l2_columns].agg(['mean', 'std'])
    
    # Optional: flatten MultiIndex columns
    grouped_stats.columns = [f"{col}_{stat}" for col, stat in grouped_stats.columns]
    
    # Display the result
    display(grouped_stats)

def top_x_percent_per_label(merged_df, percent):
    l2_columns = [col for col in merged_df.columns if col.startswith('dist_')]

    # List to store the final output
    top_10_percent_idxs = []
    quant = 1 - percent
    
    for col in l2_columns:
        top_idxs = []
        for label in merged_df['labels'].unique():
            subset = merged_df[merged_df['labels'] == label]
            threshold = subset[col].quantile(quant)
            idxs = subset[subset[col] >= threshold]['idx'].to_numpy()
            top_idxs.append(idxs)
        
        # Combine all labels' top idxs into a single array (e.g., union)
        combined = np.unique(np.concatenate(top_idxs))
        top_10_percent_idxs.append(combined)

    union_all = reduce(np.union1d, top_10_percent_idxs)
    intersection_all = reduce(np.intersect1d, top_10_percent_idxs)

    print(f'Jaccard Similarity: {len(intersection_all)/len(union_all)}')
    print(f'Size of Union: {len(union_all)}')
    print(f'Size of Intersection: {len(intersection_all)}')
    
    return top_10_percent_idxs, union_all, intersection_all

def bottom_x_percent_per_label(merged_df, percent):
    l2_columns = [col for col in merged_df.columns if col.startswith('dist_')]

    # List to store the final output
    bottom_10_percent_idxs = []
    quant = 1 - percent
    
    for col in l2_columns:
        bottom_idxs = []
        for label in merged_df['labels'].unique():
            subset = merged_df[merged_df['labels'] == label]
            threshold = subset[col].quantile(percent)
            idxs = subset[subset[col] <= threshold]['idx'].to_numpy()
            bottom_idxs.append(idxs)
        
        # Combine all labels' top idxs into a single array (e.g., union)
        combined = np.unique(np.concatenate(bottom_idxs))
        bottom_10_percent_idxs.append(combined)

    union_all = reduce(np.union1d, bottom_10_percent_idxs)
    intersection_all = reduce(np.intersect1d, bottom_10_percent_idxs)

    print(f'Jaccard Similarity: {len(intersection_all)/len(union_all)}')
    print(f'Size of Union: {len(union_all)}')
    print(f'Size of Intersection: {len(intersection_all)}')
    
    return bottom_10_percent_idxs, union_all, intersection_all