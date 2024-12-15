# Eigenvector-like centrality in non-uniform Hypergraphs

This repository contains the scripts, data and (pre-processed) figures from the : Uplifting edges in higher order networks: spectral
centralities in non-uniform hypergraphs" paper by G. Contreras-Aso, C. Pérez-Corral, M. Romance, published in [*AIMS Mathematics*](https://doi.org/10.3934/math.20241539) and available as a preprint at [arXiv:2310.20335](https://arxiv.org/abs/2310.20335).


## Structure of the repository

The following files and directories are briefly summarized, in the same order as they appear within the paper.

- `hyperfunctions.py`: Our proposed measures and several auxiliary functions are defined here.
- `other_measures.py`: The vector centrality and the blowup uniformization (see paper for definitions), used for numerical comparisons, are defined here.
- `Pairwise_comparisons.ipynb`: Uplift from an example pairwise graph to a 3-uniform hypergraph whose HEC is calculated, and compared with the original EC.
- `UPHEC_example.ipynb:`: This notebook contains the computation of the UPHEC at each order on the toy model hypergraph, as well as the vector centrality on it.
- `RankingAnalysis.ipynb`: This notebook analyzes the output data from the previous notebook, in particular computing Kendall tau correlations between rankings. 
- `ZEC_example.ipynb`: This notebook contains the computation of the Perron-like Z-eigenvector of an example hypergraph, following the example in the main text.
- `Examples/`: contains several notebooks and a subfolder
  - `Examples/{hypergraph}_example.ipynb`: These notebooks compute, for each hypergraph, all of the centralities (UPHEC, HEC, blowup uniformization) and saves them as a .csv file in the `Output/` folder.
  - `Examples/RankingAnalysis/{hypergraph)_RankingAnalysis.ipynb`: This folder contains notebooks analyzing the already compued centralities for each hypergraph, generating the Kendall-tau plots to be save in `Figures/`.
- `Output/`: Output of the `Examples/{hypergraph}_example.ipynb` notebooks.
- `Figures/`: Pre-processed plots, saved from `Examples/RankingAnalysis/{hypergraph)_RankingAnalysis.ipynb`.


## Data sources

All real hypergraphs are drawn from the XGI library [1] hypergraph database (XGI-DATA). There are some synthetic ones which we generate and analyze in the `UPHEC_example.ipynb` and `ZEC_example.ipynb`. Lastly, we also generate several random ones in `Examples/random_example.ipynb` with the `xgi.generators.random.random_hypergraph()` function.

[1] Landry, N. W., Lucas, M., Iacopini, I., Petri, G., Schwarze, A., Patania, A., & Torres, L. (2023). XGI: A Python package for higher-order interaction networks. Journal of Open Source Software, 8(85), 5162. https://doi.org/10.21105/joss.05162


## How to cite

Bibtex citation:
```
@article{contreras2023uplifting,
  title = {Uplifting edges in higher-order networks: Spectral centralities for non-uniform hypergraphs},
  journal = {AIMS Mathematics},
  volume = {9},
  number = {11},
  pages = {32045-32075},
  year = {2024},
  issn = {2473-6988},
  doi = {10.3934/math.20241539},
  url = {https://www.aimspress.com/article/doi/10.3934/math.20241539},
  author = {Gonzalo Contreras-Aso and Cristian Pérez-Corral and Miguel Romance},
  keywords = {graph theory, hypergraphs, centrality, hypermatrices, eigenvectors},
}
```
