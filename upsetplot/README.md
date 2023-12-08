# About

This folder comprises essential code and data required to produce an [UpSet plot](https://upset.app/) representing error labels in the [TPME.csv](https://github.com/AudayBerro/TPME/blob/master/TPME.csv) dataset. The visualization is generated using the  [Intervene software](https://asntech.shinyapps.io/intervene/).


### What is an UpSet plot?
An [UpSet plot](https://upset.app/) is a data visualization technique used to visualize the intersections of sets in a data set. It is particularly useful for analyzing the overlap and unique elements among multiple sets. The plot typically consists of a matrix of squares, where each square represents a set, and the intersections are filled to indicate the presence of elements in those intersections.

The UpSet plot visually depicts the intersection of error labels within the TPME.csv dataset, offering insights into the relationships and overlaps between different error categories.

### Intervene Software

The [Intervene software](https://asntech.shinyapps.io/intervene/) is employed to create the UpSet plot. It provides an interactive and user-friendly platform for the analysis and visualization of set intersections, facilitating a comprehensive understanding of data overlaps.

Feel free to explore and utilize the provided code and data to generate your UpSet plot and gain valuable insights into the error labeling patterns in the TPME.csv dataset.

## Table of Contents
- 📁 UpsetBinaryForamtDataForInterven
- 📁 UpsetPlotWithIntervene
- 📝 TPME_labels_only.txt
- 📝 errors_in_order.txt
- 🐍 compute_error_rate_order.py
- 🐍 convert_upset_format.py
- 📊 upset_data.csv


## Generating UpSet Plot for TPME using Intervene

To obtain the [UpSet_plot_Intervene.pdf](https://github.com/AudayBerro/TPME/blob/master/UpSet_plot_Intervene.pdf), use the `convert_upset_format.py` script provided in this repository. The script is designed to transform the *TPME_labels_only.txt* dataset, which exclusively contains error labels extracted from the *error_category* columns in the [TPME.csv](https://github.com/AudayBerro/TPME/blob/master/TPME.csv) dataset.

The purpose of `convert_upset_format.py` is to prepare the data in a suitable format for visualization on the [Intervene](https://asntech.shinyapps.io/intervene/) platform.

The script will generate plots for the entire TPME dataset.

**Usage:**

```bash
python convert_upset_format.py  -r  TPME_labels_only.txt
```

### Visualizing UpSet Plot on the Intervene Platform

After executing the script, the result will be the `upset_data.csv` file. Follow these steps to visualize the UpSet plot on the [Intervene platform](https://asntech.shinyapps.io/intervene/):

1. Visit the [Intervene platform](https://asntech.shinyapps.io/intervene/).
2. Click on the **UpSet** option located in the Dashboard's top-left corner. You can also use [this link](https://asntech.shinyapps.io/intervene/_w_8f88f930/#shiny-tab-upset).
3. In the Upload tab, select the **Binary data (0 & 1)** option.
4. Upload the generated `upset_data.csv` file to the platform.

Upon successful upload, the plot visualization will appear on the right half of the platform.

### Adding More Intersections

If you wish to include additional intersections, feel free to customize the values in the **Settings** tab:

- **Select sets:** Choose the error labels you want to display in the plot.
- **Number of intersections to show:** Increasing this number will reveal more label intersections in the plot.

Explore the various settings to tailor the visualization according to your analysis needs.
