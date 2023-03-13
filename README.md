# Eigenvector-like Centrality in non-uniform Hypergraphs

Lista de cosas por hacer:

## Implementación del algoritmo

- Función para crear el tensor de adyacencia a partir de hipergrafos XGI. 
  - [x] Función base
  - [x] Uplift (con el nodo "fantasma")
  - [x] Projection (dividiendo hiperaristas de mayor dimension)

- [x] HEC a mano a partir del tensor de adyacencia (basado en la implementación de XGI), incluyendo posibles nodos extra

- [x] Función para crear el hipergrafo k-uniforme a partir del hipergrafo k-no-uniforme


# Comparación con algoritmos ya existentes

- Comparación con red púramente pairwise
  - [x] Pruebas preliminares y consistencia
  - [x] Explicación de los posibles cruces entre EIG y HEC
  - [x] Sampleo masivo de redes aleatorias, calculando el Kendall's tau y/o Spearman's rho

- Nosotros VS Benson en grafos k-uniformes, usando las ya implementadas en XGI: 
  - [ ] Clique
  - [ ] ZEC
  - [ ] HEC (como media ponderada de todas las dimensiones??)

  - Su idea de añadir no-uniformidad repitiendo índices (pág. 17)
    - [x] Función base
    - [ ] Comparación con la nuestra

- Nosotros VS (Tudisco & Higham), usando la implementada en XGI:
  - [x] Leer bien lo que proponen
  - [ ] Encontrar funciones f, g, phi, psi apropiadas con las que comparar nuestro método

- Otras medidas no basadas en HEC, ZEC:
  - Vector centrality
    - [x] Función base
    - [ ] Comparación con uplift + projection para cada dimensión
  - [ ] Otras? (hipergrado, hiperbetweenness...)

## Datasets

- [x] Datasets "reales", sacados de algún repositorio o webs como la de Benson.
- Generador de hipergrafos aleatorios de distinto tipo.
  - [ ] Usando alguno ya existente, como los del paquete XGI.
  - [ ] Creándolos a partir de un grafo estándar, haciendo que aristas se conviertan en hiperaristas juntando nodos en ellas aleatoriamente.
  - [ ] Otra forma?