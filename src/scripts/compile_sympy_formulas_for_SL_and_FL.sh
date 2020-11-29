# call with the constraint name like ./compile_sympy_formulas_for_SL_and_FL.sh pipes

# SL
cd ../semantic_loss
echo -e "\nCompiling SL constriant to obtain the SDD and the VTREE"
python constraints_to_cnf.py -i ../in/semantic_loss_constraints/constraints_as_propositional_logic/${1}.sympy -o tmp.txt 
pysdd -c tmp.txt -W ../in/semantic_loss_constraints/constraints_as_vtree/${1}.vtree  \
    -R ../in/semantic_loss_constraints/constraints_as_sdd/${1}.sdd
rm tmp.txt

# FL
cd ../fuzzy
echo -e "\nCompiling FL constriant to obtain the .fuzzy pickle"
python constraints_to_sympy_tree.py -i ../in/semantic_loss_constraints/constraints_as_propositional_logic/${1}.sympy \
    -o ../in/semantic_loss_constraints/constraints_as_sympy_tree

cd ../scripts
echo "All done for constriant ${1} !"