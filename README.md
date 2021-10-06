# EUFSFC:
 
This project aims to perform an efficient feature selection on high-dimensional datasets through feature clustering. Two high-dimensional datasets are included here.

For continuous feature selection, please use the FC_1.py or FCpse.py files. 
(Note: the FC_1.py file implements the same cluster merge procedure from discrete feature cluster analysis and it is much faster than the FCpse.py.
The FCpse.py file utilizes the pseudo feature generation procedure and it took longer running time.)

For discrete feature selection, please download the FC_2.py file with the entropy_estimators.py file.
(Note: the entropy_estimators.py function is modified from the Greg Ver Steeg and the references are provided below:
1.A Kraskov, H Steegbauer, P Grassberger. Estimating Mutual Information PRE 2004.
2.Greg Ver Steeg and Aram Galstyan Information-Theoretic Measures of Influence Based on Content Dynamics. WSDM, 2013.
3.Greg Ver Steeg and Aram Galstyan Information Transfer in Social Media. WWW, 2012.)

Please cite the following article for any use or modifications of this project.

Yan, X., Nazmi, S., Erol, B. A., Homaifar, A., Gebru, B., & Tunstel, E. (2020). An efficient unsupervised feature selection procedure through feature clustering. Pattern Recognition Letters, 131, 277-284.
