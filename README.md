# Eigenvector-like centrality in non-uniform Hypergraphs

This repository contains the scripts, data and (pre-processed) figures from the Uplifting edges in higher order networks: spectral
centralities in non-uniform hypergraphs" paper by G. Contreras-Aso, C. Pérez-Corral, M. Romance, available as a preprint at [Arxiv](????).


## Structure of the repository

The following files and directories are briefly summarized, in the same order as they appear within the paper.

- `hyperfunctions.py`: Our proposed measures and several auxiliary functions are defined here.
- `other_measures.py`: The vector centrality and alternative uniformization (see paper for definitions), used for comparisons, are defined here.
- `Pairwise_comparisons.ipynb`: Uplift from an example pairwise graph to a 3-uniform hypergraph whose HEC is calculated, and compared with the original EC.
- `UPHEC_example.ipynb:`: This notebook contains the computation of the UPHEC at each order on the toy model hypergraph, as well as the vector centrality on it.
- `TagsAskUbuntu_example.ipynb`: This notebook computes all of the centralities (UPHEC, HEC, Alternative uniformization) which will later be compared, on the Tags_ask_ubuntu dataset.
- `RankingAnalysis.ipynb`: This notebook analyzes the output data from the previous notebook, in particular computing Kendall tau correlations between rankings. 
- `ZEC_example.ipynb`: This notebook contains the computation of the Perron-like Z-eigenvector of an example hypergraph, following the example in the main text.
- `Dataset/` directory: contains the Tags_ask_ubuntu dataset [1].
- `Output/`: Output of the `TagsAskUbuntu_example.ipynb` notebook.
- `Figures/`: Pre-processed plots.


[1] "Simplicial closure and higher-order link prediction". Austin R. Benson, Rediet Abebe, Michael T. Schaub, Ali Jadbabaie, and Jon Kleinberg. PNAS vol. 115, no. 48
 `