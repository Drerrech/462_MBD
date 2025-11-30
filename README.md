# MBD and BB toolbox
## this repository includes:
1) python code for calling different variations of the MBD algorithm, with 4 problems supplied (note that the executables were compiled for an M4 mac)
2) python code for plotting the results produced by the algorithms  

note: the majority of useful files are in the 'project' folder  
## how to call the MBD:
the 'mbd.py' file contains two versions of the algorithm, mbd_basic and mbd_v2, the diffrences are in how delta is updated and how the descent direction is selected.

calling mbd_basic on a plain function of rheology:  
```mbd.mbd_basic(A1_rheology.rheology_4_sum, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="something/something.txt")```

calling with sum-of-models property:  
```mbd.mbd_basic(A1_rheology.rheology_4_element_wise, x, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path="something/something.txt", f_post_process=A1_rheology.rheology_post_processing)```  
**note the f_post_process function supplied to the method**

## how to plot results:

For plotting, see the mbd_plots script.

Each method in this script has a Doc String describing the purpose and the function of each method.

There are four python notebooks in the project folder, named rheology.ipynb, rungeKutta.ipynb, planeWing.ipynb and styrene.ipynb.

Each one of these contain the code to plot the performance profiles related to each of those problems. If the code is run as it is, all plots for every MBD variant will be plotted (looks like spaghetti).

The contents of each notebook cell can be pasted into a python script and run separately if one wishes to plot without using python notebooks. 

These instructions can be found at the top of the files where they may be needed. 

## useful files, described:
* mbd.py - contains the methods for calling the actual MBD algorithms, as well as a simple function hashing class (for evaluation reuse)
* models.py - functions for approximating the gradient
* line_search.py - 3 line search algorithms (1 forward backward, 2 versions of quadratic)
* 'problem_name'.py and 'problem_name'_test.py - files implementing the BB as well as generating the data

## contact
Contact Justin Dilabio (jdilabio123@gmail.com) or Karim Akhtiamov (karim.akhtyamov@gmail.com) if you have any questions about the code.
