# Eigenvector-like Centrality in non-uniform Hypergraphs

Lista de cosas por hacer:

## Implementación del algoritmo

- [ ] Función para crear el tensor de adyacencia. Importante: factores combinatorios.
- [ ] Comparar si manualmente es más rápido calcular este HEC que el de XGI
- [ ] Función para crear el hipergrafo k-uniforme a partir del hipergrafo k-no-uniforme.


# Comparación con algoritmos ya existentes

- Comparación con red púramente pairwise

- Nosotros VS Benson, usando las ya implementadas en XGI: 
  - [ ] Clique
  - [ ] ZEC
  - [ ] HEC (vectorial, para distintas dimensiones k)
  - [ ] HEC (como media ponderada de todas las dimensiones)

  - [ ] Su idea de añadir no-uniformidad repitiendo índices (pág. 17)

- Nosotros VS (Tudisco & Higham):
  - [ ] Leer bien lo que proponen
  - [ ] Encontrar funciones f, g, phi, psi apropiadas con las que comparar nuestro método

- Otras medidas no basadas en HEC, ZEC:
  - [ ] Vector centrality
  - [ ] Otras? (hipergrado, hiperbetweenness...)

## Datasets

- [ ] Datasets "reales", sacados de algún repositorio o webs como la de Benson.
- [ ] Generador de hipergrafos aleatorios. Posiblemente usando alguno ya existente, como los del paquete XGI. Alternativamente, creándolos a partir de un grafo estándar, haciendo que aristas se conviertan en hiperaristas juntando nodos en ellas aleatoriamente.
