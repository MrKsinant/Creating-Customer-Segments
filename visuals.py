#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#############################################################
# PROGRAMMER: Pierre-Antoine Ksinant                        #
# DATE CREATED: 18/03/2019                                  #
# REVISED DATE: -                                           #
# PURPOSE: This file consists of a library of visualization #
#          functions                                        #
#############################################################


##################
# Needed imports #
##################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import warnings
from IPython import get_ipython


##########################
# Matplotlib adjustments #
##########################

# Suppress matplotlib user warnings:
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Display inline matplotlib plots with IPython:
get_ipython().run_line_magic("matplotlib", "inline")

# Use retina matplotlib backend with IPython:
get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")


########################
# Function pca_results #
########################

def pca_results(good_data, pca):
    """
    Create a DataFrame of the PCA results
    Include dimension feature weights and explained variance
    Visualize the PCA results
    """

    # Dimension indexing:
    dimensions = ["Dimension {}".format(i) for i in range(1, len(pca.components_) + 1)]

    # PCA components:
    components = pd.DataFrame(np.round(pca.components_, 4), columns=list(good_data.keys()))
    components.index = dimensions

    # PCA explained variance:
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=["Explained Variance"])
    variance_ratios.index = dimensions

    # Create a bar plot visualization:
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the feature weights as a function of the components:
    components.plot(ax=ax, kind="bar")
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios:
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i - 0.4,
                ax.get_ylim()[1] + 0.05,
                "Explained Variance\n          %.4f"%(ev))

    # Return a concatenated DataFrame:
    return pd.concat([variance_ratios, components], axis=1)


############################
# Function cluster_results #
############################

def cluster_results(reduced_data, preds, centers, pca_samples):
    """
    Visualize PCA-reduced cluster data in two dimensions
    Add cues for cluster centers and selected sample data
    """

    # Prepare data:
    predictions = pd.DataFrame(preds, columns=["Cluster"])
    plot_data = pd.concat([predictions, reduced_data], axis=1)

    # Generate cluster plot:
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map:
    cmap = cm.get_cmap("gist_rainbow")

    # Color the points based on assigned cluster:
    for i, cluster in plot_data.groupby("Cluster"):   
        cluster.plot(ax=ax,
                     kind="scatter",
                     x="Dimension 1",
                     y="Dimension 2",
                     color=cmap((i)*1./(len(centers) - 1)),
                     label="Cluster %i"%(i),
                     s=30)

    # Plot centers with indicators:
    for i, c in enumerate(centers):
        ax.scatter(x=c[0],
                   y=c[1],
                   color="white",
                   edgecolors="black",
                   alpha=1,
                   linewidth=2,
                   marker="o",
                   s=200)
        ax.scatter(x=c[0],
                   y=c[1],
                   marker="$%d$"%(i),
                   alpha=1,
                   s=100)

    # Plot transformed sample points: 
    ax.scatter(x=pca_samples[:,0],
               y=pca_samples[:,1],
               s=150,
               linewidth=4,
               color="black",
               marker="x")

    # Set plot title:
    ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross")


###################
# Function biplot #
###################

def biplot(good_data, reduced_data, pca):
    """
    Produce a biplot that shows:
    - a scatterplot of the reduced data
    - the projections of the original features
    
    Parameters
     good_data - original data, before transformation
                 (needs to be a pandas dataframe with valid column names)
     reduced_data - reduced data
                    (the first two dimensions are plotted)
     pca - pca object that contains the components_ attribute
    Return
     a matplotlib AxesSubplot object
     (for any additional customization)
    
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    """

    # Define figure:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Scatterplot of reduced data:    
    ax.scatter(x=reduced_data.loc[:, "Dimension 1"],
               y=reduced_data.loc[:, "Dimension 2"],
               facecolors="b",
               edgecolors="b",
               s=70,
               alpha=0.5)
    
    # Define feature vectors:
    feature_vectors = pca.components_.T

    # Scale factors to make the arrows easier to see:
    arrow_size, text_pos = 7., 8.

    # Projections of original features:
    for i, v in enumerate(feature_vectors):
        ax.arrow(0,
                 0,
                 arrow_size*v[0],
                 arrow_size*v[1],
                 head_width=0.2,
                 head_length=0.2,
                 linewidth=2,
                 color="red")
        ax.text(v[0]*text_pos,
                v[1]*text_pos,
                good_data.columns[i],
                color="black",
                ha="center",
                va="center",
                fontsize=18)
    
    # Set figure labels:
    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC Plane with Original Feature Projections.", fontsize=16)
    
    # Return matplotlib AxesSubplot object:
    return ax
    

############################
# Function channel_results #
############################

def channel_results(reduced_data, outliers, pca_samples):
    """
    Visualize PCA-reduced cluster data in two dimensions using the full dataset
    Data is labeled by "Channel" and cues added for selected sample data
    """

    # Check that dataset is loadable:
    try:
        full_data = pd.read_csv("customers.csv")
    except:
        print("=> Dataset could not be loaded: 'customers.csv' is missing.")       
        return False

    # Create the channel DataFrame:
    channel = pd.DataFrame(full_data["Channel"], columns=["Channel"])
    channel = channel.drop(channel.index[outliers]).reset_index(drop=True)
    labeled = pd.concat([reduced_data, channel], axis=1)

    # Generate the cluster plot:
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map:
    cmap = cm.get_cmap("gist_rainbow")

    # Color the points based on assigned channel:
    labels = ["Hotel/Restaurant/Cafe", "Retailer"]
    grouped = labeled.groupby("Channel")
    for i, channel in grouped:   
        channel.plot(ax=ax,
                     kind ="scatter",
                     x="Dimension 1",
                     y="Dimension 2",
                     color=cmap((i - 1)*1./2),
                     label=labels[i - 1],
                     s=30)

    # Plot transformed sample points:   
    for i, sample in enumerate(pca_samples):
        ax.scatter(x=sample[0],
                   y=sample[1],
                   s=200,
                   linewidth=3,
                   color="black",
                   marker="o",
                   facecolors="none")
        ax.scatter(x=sample[0] + 0.25,
                   y=sample[1] + 0.3,
                   marker="$%d$"%(i),
                   alpha=1,
                   s=125)

    # Set plot title:
    ax.set_title("PCA-Reduced Data Labeled by 'Channel'\nTransformed Sample Data Circled")