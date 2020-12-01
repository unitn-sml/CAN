import argparse
import numpy as np

from plot_utils import exp_translate_name, get_columns_from_event, get_experiment_events, \
    correct_for_weights, plot_column, get_exp_name_with_no_run


def main(columns_to_plot, std_runs):
    # each plot is a list of data from different experiments
    data_to_plot = {name: {} for name in columns_to_plot}

    exp_names, event_files = get_experiment_events(path="../../level_generation/out/tensorboard/mario-all-8layers-deep")

    # open each event file and get the data we need
    for exp, event in zip(exp_names, event_files):
        columns_data = get_columns_from_event(exp, event, columns_to_plot)
        for col, data in zip(columns_to_plot, columns_data):
            assert exp not in data_to_plot[col]
            data_to_plot[col][exp] = data

    if std_runs:
        # compute std over different runs, plot both mean and std

        # remove "run" from experiment names, keep the set of it
        name_to_original_name_mapping = {name: get_exp_name_with_no_run(name) for name in exp_names}
        exp_names = set([v for k, v in name_to_original_name_mapping.items()])

        # new data to plot is a concatenation of data from different runs of the same experiment
        tmp_data_to_plot = {name: {} for name in columns_to_plot}

        for col in columns_to_plot:
            for name in exp_names:
                runs = [k for k, v in name_to_original_name_mapping.items() if v == name]
                data = [data_to_plot[col][run_name] for run_name in runs]
                mshape = min([d.shape[0] for d in data])
                data = [d[:mshape] for d in data]
                data = np.stack(data, axis=1)
                tmp_data_to_plot[col][name] = data
        data_to_plot = tmp_data_to_plot

    formatted_exp_names = [exp_translate_name(name) for name in exp_names]
    for col in columns_to_plot:
        data = [data_to_plot[col][exp] for exp in exp_names]
        plot_column(col, formatted_exp_names, data, std_runs=std_runs)
        # plot_column(col, formatted_exp_names, data, s=0.07, std_runs=std_runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--std_runs",
                        help="If to check for different runs of the same experiment and plot also the std",
                        action="store_true",
                        required=False, default=False)
    args = parser.parse_args()

    # what to plot
    columns_to_plot = [
        # "can/generator_3/pc_custom_Validity",
        # "can/generator_3/pc_custom_Novelty",
        # "can/generator_3/pc_custom_Uniqueness",
        # "can/generator_3/pc_custom_full_Validity",
        # "can/generator_3/pc_custom_full_Novelty",
        # "can/generator_3/pc_custom_full_Uniqueness",
        "can/generator_3/pc_custom_rows_Validity",
        "can/generator_3/pc_custom_rows_Novelty",
        "can/generator_3/pc_custom_rows_Uniqueness",
        ]

    main(columns_to_plot, args.std_runs)
