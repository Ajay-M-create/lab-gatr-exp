# Experiments: Analyzing the Performance Divergence Between GATr and Transformers Across Task Complexity

## Introduction

In the realm of 3D point cloud analysis, understanding the strengths and limitations of different neural network architectures is pivotal. This series of experiments aims to investigate how the performance of **Geometric Algebra Transformers (GATr)** and **standard Transformers** diverges as the complexity of the surrogate tasks increases. By systematically scaling task complexity, we seek to uncover the scenarios where each model excels or underperforms, providing valuable insights for future model selection and architectural improvements.

## Objectives

- **Evaluate GATr and Transformer Performance:** Assess how both models perform on various surrogate tasks ranging from simple to highly complex.
  
- **Understand Complexity Impact:** Determine the relationship between task complexity and model efficacy, identifying at which points the models’ performances begin to diverge significantly.
  
- **Provide Insights for Model Selection:** Offer guidance on selecting the appropriate model based on the complexity of the task at hand.

## Models Under Investigation

### Geometric Algebra Transformers (GATr)

GATr leverages the principles of Geometric Algebra to process and interpret geometric data effectively. It integrates geometric transformations directly into the transformer architecture, enhancing its ability to handle spatial relationships inherent in 3D point clouds.

### Standard Transformers

Standard Transformers, renowned for their success in natural language processing, are adapted here for 3D point cloud tasks without specialized geometric enhancements. Their performance serves as a baseline to evaluate the added value of geometric algebra in complex tasks.

## Surrogate Tasks and Complexity Scaling

To evaluate the models comprehensively, a suite of surrogate tasks with varying levels of complexity has been designed:


1. **Number of Vertices (`num_vertices`):**  
   - *Description:* Estimating the number of vertices forming the convex hull of the point cloud (should always be 5 for this dataset).
   - *Complexity Level:* Very Easy 
   - *File:* `GATr_num_vertices.py`, `transformer_num_vertices.py`

2. **Number of Facets (`num_facets`):**  
   - *Description:* Predicting the number of facets in a convex hull derived from a 3D point cloud.
   - *Complexity Level:* Easy
   - *File:* `GATr_num_facets.py`, `transformer_num_facets.py`


3. **Surface Area (`surface_area`):**  
   - *Description:* Calculating the surface area of the convex hull encompassing the point cloud.
   - *Complexity Level:* Medium 
   - *File:* `GATr_surface_area.py`, `transformer_surface_area.py`

4. **Volume (`volume`):**  
   - *Description:* Determining the volume enclosed by the convex hull of the point cloud.
   - *Complexity Level:* High  
   - *File:* `GATr_volume.py`, `transformer_volume.py`

5. **Inradius (`inradius`):**  
   - *Description:* Computing the inradius of the convex hull, which is the radius of the largest sphere inscribed within the hull.
   - *Complexity Level:* Very High  
   - *File:* `GATr_inradius.py`, `transformer_inradius.py`

Each task incrementally introduces more complex geometric computations, thereby increasing the overall difficulty for the models to achieve accurate predictions.

## Experimental Setup

### Dataset

All experiments utilize a standardized dataset of 3D point clouds stored in the `3d_point_cloud_dataset` directory. Each point cloud is represented in `.txt` files with corresponding labels for the target quantities (`vertices`, `facets`, `area`, `volume`, `inradius`).

### Data Preparation

For each surrogate task, the `ConvexHullDataset` class is instantiated with the appropriate target parameter. The dataset is split into training and testing sets with an 80-20 ratio for  evaluation.

### Model Training

Both **GATr** and **Transformer** models are trained using their respective Python scripts. The training processes involve:

- **Batch Size:** 1  
- **Epochs:** 50  
- **Learning Rate:**  
  - GATr: `1e-4`  
  - Transformer: `1e-4`  
- **Loss Function:**  
  - GATr: `L1Loss`  
  - Transformer: `L1Loss`  
- **Optimizer:** Adam  
- **Gradient Clipping:** Applied in GATr to prevent exploding gradients.

### Evaluation Metrics

The primary metric for evaluation is the **Mean Absolute Error (MAE)** between the predicted and actual target values. Training and testing losses are recorded and plotted to visualize the learning curves over epochs.

## Execution and File Structure

Each task has dedicated scripts for both models, following a consistent naming convention:

- **GATr Scripts:**
  - `GATr_<task>.py`  
    Examples: `GATr_num_facets.py`, `GATr_volume.py`

- **Transformer Scripts:**
  - `transformer_<task>.py`  
    Examples: `transformer_surface_area.py`, `transformer_inradius.py`

These scripts encompass data loading, model initialization, training loops, and loss curve plotting. For instance, `GATr_num_facets.py` handles the task of predicting the number of facets using the GATr model, while `transformer_num_facets.py` performs the same task using a standard Transformer.

## Results Visualization

Post-training, loss curves for both models across all tasks are saved in the `./loss_curves` directory. These visualizations aid in comparing the convergence behavior and overall performance between GATr and Transformers as task complexity escalates.

Example Plot Titles:
- `GATR_loss_curves_num_facets.png`
- `Transformer_loss_curves_inradius.png`

## Anticipated Outcomes

Through these experiments, we hypothesize that:

- **GATr** will demonstrate superior performance on higher complexity tasks due to its inherent geometric processing capabilities.
  
- **Transformers** may perform competitively on simpler tasks but could struggle with increased complexity, leading to divergent performance trends.

## Conclusion

This experimental framework is designed to meticulously assess and compare the capabilities of GATr and standard Transformers in handling 3D geometric tasks of varying complexity. The insights gleaned will inform the development of more effective models tailored to specific geometric analysis challenges in point cloud data.
