## To create a new environment:
`conda create --name <env_name> --file requirements_conda.txt`

## To install dependencies into an existing environment (with python 3.6.13):
`pip install -r requirements_pip.txt`


## Notebooks to create a dataset of claims:
 
`lupa/SeleniumLupa.ipynb` - Extracts documents from Lupa Agency

`lupa/LupaDatasetTreatment.ipynb` - Separates claims and evidence from Lupa documents

`selenium_boatos.ipynb` - Extracts documents from Boatos.org. It is necessary to input a file with a list of urls (For example, the Fakepedia dataset)

`DatasetCreation.ipynb` - Join all datasets (Lupa, Boatos.org (fakepedia), FACTCKBR, FakeRecogna), creates samples with label NEI, and splits dataset (train, test and valid).

## To train the classifier model:

`bert_train_classifier.ipynb` - Given a dataset of claims with evidence, train a Bert classifier

Trained models must be placed in the models folder, and can be obtained here:
https://github.com/feliperibeirouff/brazilianfactchecking/releases/tag/1.0.0

## To perform the fact-checking steps in a dataset:

`pipeline.ipynb` - Executes document retrieval, evidence selection and classification

`get_metrics.ipynb` - To evaluate the performance of the process
