This folder contains the scripts to reproduce the result in the paper "Multiplexed protein profiling reveals spatial subcellular signaling networks"

Due to very large dataset size, please contact authors to get access to the data.

<!-- From IMC image dataset, we can look at Immune cell expression in health and disease  -->

<!-- ![Alt text](figures/stats/count_Icell.png)

The expression level for all the markers across health and disease can be summarized as follow:

![Alt text](figures/stats/dotplot_expression.png)

Then, pixel level clustering with KMeans is performed to extract the anatomical properties.

![Alt text](figures/clusters/DT2_cluster_by_marker.png?raw=true)

The clustered images are then combined together in one image in order to visualize the clusters representation

![Alt text](figures/clusters/DT2_cluster_combined2.png?raw=true)

In order to better understand the spatial anatomy in various dataset, intra and inter cluster distance network is generated

![Alt text](figures/clusters/DT2_cluster_inter.png)
![Alt text](figures/clusters/DT2_cluster_intra.png)

It is also possible to look at individual markers by generating spatial reference map with fixed node 

![Alt text](figures/clusters/DT2_spatial_reference.png)

It is possible to look at the 2D and 3D topographic layer of specific markers such as CD44, Pankeratin and GranzymeB

![Alt text](figures/3D_topo/DT2_expression2.png)
![Alt text](figures/3D_topo/DT2.png) -->


# Organization

## Notebooks 
"notebooks" folder contains jupyter notebook script used:
- 01_processing is the script for performing image processing step that includes background removal, z-stack best focus selection, multiplex cycle shift registration
- 02_segmentation is the script for generating segmentation mask using cellpose
- 03_extract_expression_level is the script used for extract mean intensity and area per cell from cell mask
- 04_generate_figures is the script used for generating multiplex protein marker images
- 05_cell_nuclei_intensity_plot is the script used for generating per cell and marker mean intensity variation in the spatial domain
- 06_HM_marker is the script for generating heatmap plot of marker expression level per cell
- 07_pixel_clustering is the script for generating pixel level clustering for all marker across field of view
- 08_correlation_plot is the script used for generating cell level marker intensity correlation
- 09_prediction is the script used for generating marker intensity prediction between protein markers
- 10_parallel_clustering_3D is the script for generating pixel level clustering in 3D by selecting multiple z-stack at best focus
- revision_bleaching is the script for quantifying marker bleaching efficiency by looking at both intensity level difference and fourrier domain analysis 
- cell_cylce is the script for generating cell cycle information 
- correlation_bleaching is the script for generating per cell after bleach marker intensity correaltion for validation
- clustering_benchmark is the script for generating benchmark result for clustering 
- min_intensity is the script for generating minimum intensity threshold level study 

## Source code
"src" folder contains customs scripts used:
- "io.py" is the custom python scripts for input output function
- "utils.py" is the custom python scripts for custom utility functions


