Install (maybe without conda if it does not compile):
```bash
pip install git+https://github.com/wannesm/PySDD.git#egg=PySDD
```

Use with:
```bash
pysdd -c dimacs.txt -W ../in/semantic_loss_constraints/constraints_as_sdd/pipes.sdd -W ../in/semantic_loss_constraints/constraints_as_vtree/pipes.vtree
```