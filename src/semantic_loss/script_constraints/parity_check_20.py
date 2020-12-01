"""
Hastily put togheter to generate parity check constraints to use later for semantic loss,
change n and m to your liking.

"""


def get_row_odd_states(row_index, m):
    variables = ["X.%s.%s" % (row_index, i) for i in range(1, m)]
    variables = ",".join(variables)
    xor = "Xor(%s)" % variables
    return xor


def get_column_odd_states(column_index, m):
    variables = ["X.%s.%s" % (i, column_index) for i in range(1, m)]
    variables = ",".join(variables)
    xor = "Xor(%s)" % variables
    return xor


# what to write to file
output = ["shape [20, 20]"]

# number of control pixels
n = 10
# number of pixels to check for parity for each row/column
m = 10

for i in range(1, n):
    row_check = "X.%s.0" % i
    row_OR = get_row_odd_states(i, m)
    row_check = "Equivalent(%s, %s)" % (row_check, row_OR)
    output.append(row_check)

    col_check = "X.0.%s" % i
    col_OR = get_column_odd_states(i, m)
    col_check = "Equivalent(%s, %s)" % (col_check, col_OR)
    output.append(col_check)

output = "\n".join(output)
with open("parity_check_10_10.txt", "w") as file:
    file.write(output)
