# Model Change Active Learning Paper
## (To Appear)

Python code for doing active learning in graph-based semi-supervised learning (GBSSL) paradigm. Implements testing done in paper that will soon appear and be submitted for peer-review.

## Usage

To run tests in this framework, edit ``run_experiments_binary.py`` or ``run_experiments_multiclass.py`` to 
* hard-code ``DATA_FILEPATH`` variable to specify where your ``*.npz`` file is located.
* specify which acquisition functions (with their corresponding underlying GBSSL model) to test in the list variable ``acq_models``
* possible choices in ``acq_models``:
  * __acquisitions functions__ : ``mc`` (Model Change), ``uncertainty`` (Uncertainty), ``vopt`` (VOpt), ``sopt`` (SigmaOpt), ``rand`` (Random)
  * __binary models__ : ``gr`` (Gaussian Regression), ``log`` (Logistic Loss), ``probitnorm`` (Probit - Normal)
  * __multiclass models__ : ``gr``(Gaussian Regression), ``ce`` (Cross-Entropy)
  * Separate __acquisition function__ and __model__ with double-dash: e.g. ``mc--gr`` --> Model Change acquisition function in Gaussian Regression Model.

## Results in Paper
Example plots from code in ``results/acc_figures.py``
### Multiclass Gaussian Regression
MNIST           |  Salinas A       | Urban 
:-------------------------:|:-------------------------:|:-------------------------:
![](results/gh-pics/acc-mgr-mnist.png) |  ![](results/gh-pics/acc-mgr-salinas.png) | ![](results/gh-pics/acc-mgr-urban.png)
