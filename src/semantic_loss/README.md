# Table of contents
1. [constraints_to_cnf](#tocnf)
2. [constraints generator](#cgen)
3. [py3psdd](#py3psdd)
4. [Semantic losses](#semlosses)  
4.1 [SemanticLossStatistics](#semlossstats)
5. [WMC statistics](#wmcstats)
6. [VNU statistics](#vnustats)


In the "extra" directory you will find the conda and pip list of stuff that you need and can
install automatically with conda and pip.


You have to install the py3psdd package provided in this repository by yourself.

<a name="tocnf"></a>  
## Constraints_to_cnf

##### Requirements: sympy

Constraints_to_cnf is a module which allows the writing
of constraints in propositional logic in the syntax of **sympy, and
then translate them to DIMACS**. Constraints are expressed
1 by line, and are considered to be in an "and" relationsip.

Moreover, it allows to refer to variables not just by a single
index, like Xi, but via more indexes, X.i.j.z, etc, as if
they were in a tensor of arbitrary shape.  
When this syntax
is translated to DIMACS, the indexes are converted in a single
dimension, as if the variables were in a mono dimensional
vector, while parsing, multi dimensional indexes must
respect the input shape (can't refer to variables that do
not exist, out of bounds, etc.)

By using the **sympy syntax**, constraints can now
be written with more operators:
- and (&) and or (|)
- Xor
- Nand
- Nor
- ITE, if then else
- implies, by using ">>" and "<<".
- Equivalent(X1, X2, X3), etc.
- check out https://docs.sympy.org/latest/modules/logic.html for
more alternatives to the syntax, like Or(a, b) instead of a|b.
- essentially, you are not limited to the syntax of the logic
module of sympy, but can access the whole package if you want to try
funky stuff, this is however not supported in this package and you will
probably meet unexpected behaviours, and you should
stick to logic operators. If you go looking for unexpected behaviour
you will find it.:shipit:

Example usage: let's say we have 4 variables with 3 possible
states (think of some multinomial distribution), we can imagine
our states as arranged in a tensor of shape [4,3]. We would
like to say that when the first variable assumes state 1, then
the second variable must assume state 2, moreover, the third variable
has always state 3.

Keep in mind that variables are referred starting from index 0.
```
# this is a comment
shape [4,3]

# i like blank lines


# my constraints

# var1.state1 implies var2.state2
X0.0 >> X.1.1

# var3 must always have state 3
X2.2

# given that states are mutually exclusive, we should also state that
X0.0 >> (~X0.1 & ~X0.2)
X0.1 >> (~X0.0 & ~X0.2)
X0.2 >> (~X0.0 & ~X0.1)

X1.0 >> (~X1.1 & ~X1.2)
X1.1 >> (~X1.0 & ~X1.2)
X1.2 >> (~X1.0 & ~X1.1)

X2.0 >> (~X2.1 & ~X2.2)
X2.1 >> (~X2.0 & ~X2.2)
X2.2 >> (~X2.0 & ~X2.1)

X3.0 >> (~X3.1 & ~X3.2)
X3.1 >> (~X3.0 & ~X3.2)
X3.2 >> (~X3.0 & ~X3.1)

# we should also state that variables must have at least 1 state
(X0.0 | X0.1 | X0.2)
(X1.0 | X1.1 | X1.2)
(X2.0 | X2.1 | X2.2)
(X3.0 | X3.1 | X3.2)
```

After writing this input file, you can simply call
the script.
```bash
python constraints_to_cnf.py -i myinputfile.txt -o dimacs.txt 
```
The result would be the following DIMACS file:
```
c This file was generated with the constraints_to_cnf module in this project.
c Starting from file 'example.sympy'.
c There are 13 variables present in the constraints, and 12 total variables, given by the shape [4, 3].
c
p cnf 12 18
9 0
5 -1 0
1 2 3 0
4 5 6 0
7 8 9 0
10 11 12 0
-1 -2 0
-1 -3 0
-2 -3 0
-4 -5 0
-4 -6 0
-5 -6 0
-7 -8 0
-7 -9 0
-8 -9 0
-10 -11 0
-10 -12 0
-11 -12 0
```

###### Note that DIMACS refers to variables by starting from index 1, and not 0.

Note that "-p" is an optional argument to also specify
the number of processes to use while using sympy to parse
our constraints. This might be necessary if you have many constraints,
given that sympy seems to really take a hit when parsing long strings.
While parsing many constraints can be more or less helped by
adding processes, very long constraints on single lines
will slow down the process and it might be smarter to
put them to cnf and then set them 1 per line.

Note that you can omit the dot for the first index, for
better readability; meaning that X1.2 is equal to writing
X.1.2, or X1 is the same as X.1.


More complex shapes can be as easily used, i.e. [3,4,50,200,2] etc.,
finding use cases for this is left to the reader.

**caveat**: Evaluation from sympy is turned off during
parsing, meaning that you can write down constraints
that are False, like Equivalent(X0, ~X0), or having
X0 and ~X0 on different lines.
However, I have noticed that even without evaluation
there seems to be the chance of sympy evaluating
something directly to False, which would result
in having a single constraint in the DIMACS output file, "False".
However "False" is not part of the DIMACS syntax so it will
result in an error if you try to use this output with pysdd.

Tests are in the test directory (might take some time depending
on your computer).


<a name="cgen"></a>  
## constraints_generator:

##### Requirements: sympy

Generate constraints and sample models.
Generate a directory containing 3 files:
- \<name>.config, the args that generated this directory
- \<name>.sympy constraints written as a CNF, 1 per line, sympy syntax, that can be translated to dimacs using the
    constraints_to_cnf.py script in this module
- \<name>.csv sampled models that satisfy the constraints, in a csv format, 1 model per line
    
Arguments:
- -c = OR clauses in the cnf constraints, (might actually result in less than this if randomly the same clause is 
generated.
- -v = Number of variables we are working on/that we are considering. Clauses will be randomly created by randomly
Oring random variables (after possibly and randomly using the not operator on them).
- -m = Max number of literals per clause, going randomly from 1 to m.
- --nmodels= Number of models to sample, might result in less if there are less than --nmodels models.
- -o = Output name for this run, the directory will have this name, and it will be used as the name of the output f
files.
- -s = Seed to use for random, 1337 by default.
- --verbose = Be verbose, might be useful in case you want to see the progress on generating clauses, sampling
models, etc.
- --simply = Try to simplify the cnf representing the constraints. This will probably take a lot of time, use with caution.
- --multinomial = If multinomial is provided, (>= 2), 
variables are now considered to represent different states 
related to variables in a multinomial distribution. 
Considering an array of length --variables, 
each chunk of length --multinomial will contain variables 
considered to be related, different 
states of a multinomial distribution. This implies that 
on top of the --clauses randomly generated
constraints, there will also be constraints specifying 
the fact that only 1 of those variables can
be true at each moment.
Note that, for example, providing --variables=10 and 
--multinomial=2 would result in a total
of 10*2 variables, because in this case you are 
basically saying that you want 10 multinomial
variables with 2 possible states.


<a name="py3psdd"></a>  
## py3psdd

To compile your DIMACS cnf files to vtrees and sdds, and to use make use of them
while running the main script you will need to install this module.  
Py3psdd is a rough port of pypsdd, which is originally written in python2, note that
not all functionalities have been tested, only the ones that are being used for semantic losses,
like generating the tensorflow arithmetic circuit.  
You can install py3psdd by calling 
```bash
pip install -e py3psdd --user
```

### Update: fast start

Install `PySDD`
```bash
pip install PySDD
```

Compile the `dimacs` file with
```bash
pysdd -c dimacs.txt -W constraint.vtree -R constraint.sdd
```
`py3psdd` will be used to build `sdd` + `vtree` in out training pipeline.

If there are problems with imports, compite `PySDD` from source:
```bash
pip install git+https://github.com/wannesm/PySDD.git#egg=PySDD
```

<a name="semlosses"></a>  
## Semantic Losses
In the losses module you can find a base implementation of the Semantic Loss class, 
this is, in principle, everything you may need.  
You can simply add new constraints to the repository by adding vtrees and sdds to the
in/semantic_loss_constraints/constraints_as_vtree and as_sdd directories, as long as the two
names match, (excluding the .vtree and .sdd), a semantic loss class will be automatically
created and you will be able to add the semantic loss to your experiment.  
Let's say you have placed placeholder.sdd and placeholder.vtree in the aforementioned directories,
a class named SemanticLoss_placeholder will be created, you will be able to add this loss
to your experiment simply by adding it to the generator losses:  
```bash
    "GENERATOR_LOSS": ["BganGeneratorLoss", "SemanticLoss_placeholder"] 
```
Of course you still can implement your own Semantic Losses, and they will not be overwritten
by automatically created classes if, for example, they have the same name as a class that would automatically
be created. 

<a name="semlossstats"></a>  
### SemanticLossStatistic
This class, that you can find in the statistic module of this directory, will automatically
pickup the value of all semantic losses and the related WMC values, and log them to tensorboard.
Given any number of semantic losses you are using, you can log them all by simply
adding this statistics to your generator statistics:
```bash
    "GENERATOR_STATISTICS": ["SemanticLossStatistics"]
```

<a name="wmcstats"></a>  
## WMCStatistic
You might be interested in the value of the weighted model counting obtained
by using the sdd related to a given constraint, without actually using its SemanticLoss
class.  
For each existing SemanticLoss class in this directory losses module, a WMC Statistic is automatically
created, which you can add to your experiment in order to log the WMC without adding the semantic
loss to the experiment.  
Let's say you have placed placeholder.sdd and placeholder.vtree in the aforementioned directories, as we
explaiend before, a class named SemanticLoss_placeholder will be created; this means
that a class named WMC_placeholder will also be created, and you will be able to log the WMC in tensorboard
by simply adding it to the generator statistics:  
```bash
    "GENERATOR_STATISTICS": ["WMC_placeholder"]
```
Again, you are still free to implement your own WMC class.

<a name="vnustats"></a>  
## VNUStatistic
You might be interested in logging the Validity, Novelty and Uniqueness values thorough training, 
where the Validity is the percentage of perfect items, according to a given constraint, Novelty
is the fraction of those perfect items that are not present in the dataset, and Uniqueness is
the number of unique perfect items w.r.t all those valid items.  
For each existing SemanticLoss class in this directory losses module, a VNU Statistic is automatically
created, which you can add to your experiment in order to log the Validity, Novelty, and Uniqueness without adding the semantic
loss to the experiment.  
Let's say you have placed placeholder.sdd and placeholder.vtree in the aforementioned directories, as we
explaiend before, a class named SemanticLoss_placeholder will be created; this means
that a class named VNU_placeholder will also be created, and you will be able to log the VNU stats in tensorboard
by simply adding it to the generator statistics:  
```bash
    "GENERATOR_STATISTICS": ["VNU_placeholder"]
```
Again, you are still free to implement your own VNU class.

