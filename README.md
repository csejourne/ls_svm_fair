In this project, we deal with binary classification with LS-SVM in a large-dimensional setting. We want to do a
performance analysis similar to [this article](https://arxiv.org/abs/1701.02967), but while adding fairness constraints.
We will focus on the separation property.

# Clone the environment using `conda` 
```
conda env create -n fair_ls_svm -f environment.yml
```
then 
```
conda activate fair_ls_svm
```

# Run the file
```
python run.py
```

To modify the parameters of the experiments, edit the corresponding option in the file `run.py`.
