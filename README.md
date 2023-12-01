## Getting Start
 
This project aims to perform an efficient feature selection on high-dimensional datasets through feature clustering. Two high-dimensional datasets are included here. (A more organized code implementation will come soon with a jupter notebook example)

Several observations can be made as follows:

1. As an unsupervised FS method, EUFSFC aims to solve the curse of dimensionality by reducing the redundancy among features and thus can potentially eliminate some useful features, which may not be competitive to supervised FS approaches on some datasets. Therefore, the comparison is made only with unsupervised approaches.
2. EUFSFC tends to select more features than supervised FS methods since it only considers the redundancy among features.
3. Possible extensions of EUFSFC can be made on incorporating the pseduo label information using self-supervised learning or directly including the label information into the selection module.

## Examples usage:

For continuous feature selection, please use the FC_1.py or FCpse.py files. 
(Note: the FC_1.py file implements the same cluster merge procedure from discrete feature cluster analysis and it is much faster than the FCpse.py.
The FCpse.py file utilizes the pseudo feature generation procedure and it took longer running time.)

For discrete feature selection, please download the FC_2.py file with the entropy_estimators.py file.
(Note: the entropy_estimators.py function is modified from the Greg Ver Steeg and the references are provided below:

1. A Kraskov, H Steegbauer, P Grassberger. Estimating Mutual Information PRE 2004.

2. Greg Ver Steeg and Aram Galstyan Information-Theoretic Measures of Influence Based on Content Dynamics. WSDM, 2013.

3. Greg Ver Steeg and Aram Galstyan Information Transfer in Social Media. WWW, 2012.)

Note: a jupternotebook example is also provided for the users to run the code from google colab and plese find the filename below:
* example_eufsfc.ipynb

## Dependencies:

Please install the following packages before running the python files above:
* Numpy
* Pandas
* Scikit-learn

## Citation format

Please cite the following article for any use or modifications of this project.

* Yan, X., Nazmi, S., Erol, B. A., Homaifar, A., Gebru, B., & Tunstel, E. (2020). An efficient unsupervised feature selection procedure through feature clustering. Pattern Recognition Letters, 131, 277-284.

Several follow-up research outcomes can be found below:
* Yan, X., Homaifar, A., Sarkar, M., Lartey, B., & Gupta, K. D. (2022). An Online Unsupervised Streaming Features Selection Through Dynamic Feature Clustering. IEEE Transactions on Artificial Intelligence.
* Yan, X., Sarkar, M., Gebru, B., Nazmi, S., & Homaifar, A. (2021, October). A supervised feature selection method for mixed-type data using density-based feature clustering. In 2021 IEEE International Conference on Systems, Man, and Cybernetics (SMC) (pp. 1900-1905). IEEE.
