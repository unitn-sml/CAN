import numpy as np

# what to plot
from plot_utils import exp_translate_name, get_columns_from_event, get_experiment_events, \
    correct_for_weights, compare_columns

columns_to_plot = [
    "can/generator_loss/batch-by-batch",
    "can/generator_loss_1/SemanticLoss_pc_custom",
    # "can/generator_3/pc_custom_full_Validity",
    # "can/generator_3/pc_custom_full_Novelty",
    # "can/generator_3/pc_custom_full_Uniqueness",
]

# each plot is a list of data from different experiments
data_to_plot = {name: [] for name in columns_to_plot}

exp_names, event_files = get_experiment_events()
exp_names = [exp_translate_name(name) for name in exp_names]

# open each event file and get the data we need
for exp, event in zip(exp_names, event_files):
    columns_data = get_columns_from_event(exp, event, columns_to_plot)
    columns_data = correct_for_weights(columns_data, exp)
    for col, data in zip(columns_to_plot, columns_data):
        data_to_plot[col].append(data)

# make sure all steps are the same for all data
for col in columns_to_plot:
    lens = [d.shape[0] for d in data_to_plot[col]]
    steps = max(lens)
    assert min(lens) == steps
    x = np.arange(steps)

# for col in columns_to_plot:
#     plot_column(col, exp_names, data_to_plot[col], ytop=30)
#     plot_column(col, exp_names, data_to_plot[col], s=0.05, ytop=30)

c1 = columns_to_plot[0]
c2 = columns_to_plot[1]
compare_columns(c1, c2, exp_names, data_to_plot[c1], data_to_plot[c2], ytop=30)
