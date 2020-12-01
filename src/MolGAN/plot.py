import pandas as pd
import argparse
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


"""
DISCLAIMER 
this is a quickly and poorly written script, do not use
it to run a nuclear facility
"""


past_limit = 150
add_paper = True
mode = "metrics"
paper_values = dict()
paper_values["valid score"] = 97.4
paper_values["unique score"] = 2.40
paper_values["diversity score"] = 91.00
paper_values["QED score"] = 47.00
paper_values["SA score"] = 84.00
paper_values["logP score"] = 65.00


def should_scale(name):
    increase = {"wmc", "SA score", "logP score", "QED score", "wmc_valence", "wmc_input_to_atom", "diversity score",
                "drugcandidate_score"}
    return name in increase or "wmc" in name


def plot_all(data, columns, title):
    for name, dataframe in data:
        for cname in columns:
            column = dataframe[cname].to_numpy()
            if should_scale(cname):
                column *= 100
            plt.plot(range(len(column)), column, label=name + cname)

    plt.legend()
    plt.ylim(0, 100)
    plt.title(title)
    plt.show()


def get_exp_name(name):
    for i in range(10):
        name = name.replace(str(i), "")
    return name


def get_mean_std(columns_data):
    limit = 161
    columns_data = [c[:161] for c in columns_data if len(c) >= limit]
    if len(columns_data) == 0:
        return None, None

    columns_data = np.stack(columns_data, axis=0)
    mean = np.mean(columns_data, axis=0)
    std = np.std(columns_data, axis=0)
    return mean, std


def group_data_by_experiment(data):
    grouped_data = defaultdict(lambda: list())

    # group data related to same parameters
    for name, dataframe in data:
        name = get_exp_name(name)
        grouped_data[name].append(dataframe)
    return grouped_data


def plot_with_std(data, columns, title):
    grouped_data = group_data_by_experiment(data)
    results = []
    for column in columns:
        for name, dataframes in grouped_data.items():
            numpy_data = [dataframe[column].to_numpy() for dataframe in dataframes if column in dataframe]
            mean, std = get_mean_std(numpy_data)
            if mean is not None:
                if should_scale(column):
                    mean *= 100
                    std *= 100
                results.append((name + "_" + column, mean, std))

    for name, mean, std in results:
        x = np.arange(len(mean))

        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.1, linewidth=0)

    plt.legend()
    plt.ylim(0, 100)
    plt.title(title)
    plt.show()


def stats_all(data, columns, past_limit, title):
    if "unique score" not in columns:
        columns = columns + ["unique score"]

    results = []
    for name, dataframe in data:
        dataframe = dataframe[columns]
        dataframe = dataframe[150:]
        below_2 = dataframe.index[dataframe['unique score'] < 2.0].tolist()
        below_2 = (below_2[0] - past_limit if len(below_2) > 0 else len(dataframe))
        dataframe = dataframe[below_2 - 1:below_2]
        results.append((name, dataframe.to_numpy()))

    N = len(columns)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    plots_legend = []
    cell_text = []
    for name, data in results:
        data = data[0]
        p1 = plt.bar(ind, data, width, yerr=0., alpha=0.5)
        plots_legend.append(p1)
        cell_text.append(['%1.2f' % x for x in data])
        # print(name, data)

    the_table = plt.table(cellText=cell_text,
                          rowLabels=[d[0] for d in results],
                          colLabels=columns,
                          loc='bottom',
                          rowLoc="center",
                          cellLoc="center",
                          bbox=[0, -0.7, 0.5, 0.5]
                          )

    plt.ylabel("Score value")
    plt.title("Scores type")
    plt.xticks(ind, columns)
    plt.yticks(np.arange(70, 101, 10))
    plt.ylim(0, 100)

    plots_legend = tuple([p[0] for p in plots_legend])
    plt.legend(plots_legend, tuple([d[0] for d in results]))

    plt.subplots_adjust(left=0.3, bottom=0.4)
    plt.title(title)
    plt.show()


def stats_with_std(data, columns, past_limit, title):
    if "unique score" not in columns:
        columns = columns + ["unique score"]

    results = []
    for name, dataframe in data:
        dataframe = dataframe[columns]
        dataframe = dataframe[150:]
        if mode == "metrics":
            below_2 = dataframe.index[dataframe['unique score'] < 2.0].tolist()
        elif mode == "uniqueness":
            below_2 = dataframe.index[dataframe['valid score'] > 99.0].tolist()
        below_2 = (below_2[0] - past_limit if len(below_2) > 0 else len(dataframe))
        # print("##########")
        # print(name)
        # print(dataframe)
        if mode == "metrics":
            dataframe = dataframe[below_2 - 1:below_2]
        elif mode == "uniqueness":
            dataframe = dataframe[below_2:below_2 + 1]
        # print(dataframe)
        results.append((name, dataframe.to_numpy()))
    results = group_data_by_experiment(results)

    N = len(columns)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    plots_legend = []
    cell_text = []

    for name, data in results.items():
        data = np.concatenate(data, axis=0)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        for i, column in enumerate(columns):
            if should_scale(column):
                mean[i] *= 100
                std[i] *= 100
        p1 = plt.bar(ind, mean, width, yerr=0., alpha=0.5)
        plots_legend.append(p1)
        cell_text.append(['%1.2f std: %1.2f' % (m, s) for m, s in zip(mean, std)])

    row_labels = list(results.keys())
    if add_paper:
        row_labels.append("PAPER")
        cell_text.append([paper_values[col] for col in columns])
        p1 = plt.bar(ind, [paper_values[col] for col in columns], width, yerr=0., alpha=0.5)
        plots_legend.append(p1)

    the_table = plt.table(cellText=cell_text,
                          rowLabels=row_labels,
                          colLabels=columns,
                          loc='bottom',
                          rowLoc="center",
                          cellLoc="center",
                          bbox=[0, -0.7, 1.0, 0.5],
                          fontsize=100
                          )
    the_table.auto_set_font_size(False)

    plt.ylabel("Score value")
    plt.title("Scores type")
    plt.xticks(ind, columns)
    plt.yticks(np.arange(0, 101, 5))
    plt.ylim(0, 100)

    plots_legend = tuple([p[0] for p in plots_legend])
    plt.legend(plots_legend, tuple(row_labels), bbox_to_anchor=(0.8, 0.8))

    plt.subplots_adjust(left=0.3, bottom=0.4)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, default="", required=True)
    parser.add_argument("-c", "--columns", type=str, default="", required=True)
    parser.add_argument("--std", default=False, required=False, action="store_true")
    parser.add_argument("--stats", default=False, required=False, action="store_true")
    parser.add_argument("--title", type=str, default="", required=False)
    parser.add_argument("--paper", default=False, required=False, action="store_true")
    parser.add_argument("--mode", type=str, default="metrics", required=False)
    parser = parser.parse_args()

    directory = parser.directory
    add_paper = parser.paper
    mode = parser.mode
    columns = parser.columns.split(",")
    title = parser.title

    # read data as (exp name, pandas dataframe)
    data = []
    for fname in os.listdir(directory):
        joined_dir = os.path.join(directory, fname)
        if os.path.isdir(joined_dir) and ("evaluation.csv" in os.listdir(joined_dir)):
            data.append((fname, pd.read_csv(os.path.join(joined_dir, "evaluation.csv"))))

        if fname == "evaluation.csv":
            data.append((directory, pd.read_csv(os.path.join(directory, "evaluation.csv"))))

    if parser.std:
        plot_with_std(data, columns, title)
    else:
        plot_all(data, columns, title)

    if parser.std:
        stats_with_std(data, columns, past_limit, title)
    else:
        stats_all(data, columns, past_limit, title)
