# annotate_celltype
Integrating multiple reference datasets for cell type annotation
# Abstract
Accurate and efficient cell type annotation is essential for single-cell sequence analysis. Currently, cell type annotation using well-annotated reference datasets with powerful models has become increasingly popular. However, with the increasing amount of single-cell data, there is an urgent need to develop a novel annotation method that can integrate multiple reference datasets to improve cell type annotation performance. Since unwanted batch effects between individual reference datasets, the integration of multiple reference datasets is still an open challenge. To address this, we proposed the scMDR and scMultiR, respectively, using multi-source domain adaptation to learn pure biological signs (cell type-specific information) from multiple reference datasets and query cells. Based on the pure biological signs, scMDR and scMultiR provide the most likely cell types for the query cells. 

#Requirement:

scikit-learn 0.24.2

torch 1.10.0

python 3.6.13

#USAGE
python scMDR.py

python scMultiR.py

#Connect

If you have any questions, please contact yanliu@njust.edu.cn
