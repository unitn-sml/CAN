import argparse
import numpy as np

from plot_utils import exp_translate_name, get_columns_from_event, get_experiment_events, \
    correct_for_weights, plot_column, get_exp_name_with_no_run

def main(columns_to_plot, paths, std_runs, max_height, min_height, title_fontsize=26, legend_fontsize=24, tick_fontsize=22, image_size=(1920, 1080)):
    # each plot is a list of data from different experiments
    data_to_plot = {name: {} for name in columns_to_plot}

    exp_names, event_files = get_experiment_events(paths=paths)

    # open each event file and get the data we need
    for exp, event in zip(exp_names, event_files):
        columns_data = get_columns_from_event(exp, event, columns_to_plot) #np.array([[1,0.2], [2,.3], [3,.4], [6, .7]])
        for col, data in zip(columns_to_plot, columns_data):
            if exp not in data_to_plot[col]:
                data_to_plot[col][exp] = data
            else:
                if isinstance(data_to_plot[col][exp], list):
                    data_to_plot[col][exp].append(data)
                else:
                    data_to_plot[col][exp] = [data_to_plot[col][exp], data]

    # data_to_plot = {
    #     "can/monsters_Validity" : {
    #         "mario-8-1-monsters-semantic-loss-005-6000": [ data1, data2 ]
    #     }
    # }

    if std_runs:
        for col in data_to_plot.keys():
            for name in data_to_plot[col].keys():
                data = data_to_plot[col][name]
                mshape = min([d.shape[0] for d in data])
                data = [d[:mshape] for d in data]
                data = np.stack(data, axis=1)
                data_to_plot[col][name] = data

    formatted_exp_names = {name: exp_translate_name(name) for name in exp_names}

    for col in columns_to_plot:
        data = [data_to_plot[col][exp] for exp in data_to_plot[col].keys()]
        tmp_names = [formatted_exp_names[n] for n in data_to_plot[col].keys()]
        plot_column(col,
            tmp_names,
            data,
            std_runs=std_runs,
            ytop=max_height,
            ybot=min_height,
            title_fontsize=title_fontsize,
            legend_fontsize=legend_fontsize,
            tick_fontsize=tick_fontsize,
            image_size=image_size,
            s=0.1
        )
        # plot_column(col, formatted_exp_names, data, s=0.07, std_runs=std_runs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--columns", required=True, type=str, nargs='+',
                        help="The columns (statistics) you want to parse and plot, at least one should be provided")
    parser.add_argument("-e", "--experiment", required=True, type=str, nargs='+',
                        help="Path to the experiment from which the statistic should be taken, default=out/tensorboard")
    parser.add_argument("--std_runs", action="store_true", required=False, default=False,
                        help="If to check for different runs of the same experiment and plot also the std")
    parser.add_argument("--min_height", required=False, default=0.0, type=float,
                        help="The min y value on the plot")
    parser.add_argument("--max_height", required=False, default=1.05, type=float,
                        help="The max y value on the plot")
    parser.add_argument("--title_fontsize", required=False, default=26, type=int,
                        help="Title fontsize")
    parser.add_argument("--legend_fontsize", required=False, default=24, type=int,
                        help="Legend fontsize")
    parser.add_argument("--tick_fontsize", required=False, default=22, type=int,
                        help="Axes values fontsize")
    parser.add_argument("--plot_width", required=False, default=22, type=int,
                        help="Image width in pixels")
    parser.add_argument("--plot_height", required=False, default=22, type=int,
                        help="Image height in pixels")
    args = parser.parse_args()

    main(args.columns, args.experiment, args.std_runs, args.max_height, args.min_height, args.title_fontsize, args.legend_fontsize, args.tick_fontsize, (args.plot_width, args.plot_height))
