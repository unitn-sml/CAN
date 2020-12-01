import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import splrep, splev
from tensorboard.backend.event_processing import event_accumulator


def col_translate_name(name):
    """
    Something quickly put togheter to format column names.
    :param name:
    :return:
    """
    if "WMC" in name:
        return "Weighted Model Count"
    elif "Validity" in name:
        return "Validity"
    elif "Novelty" in name:
        return "Novelty"
    elif "Uniqueness" in name:
        return "Uniqueness"
    elif "batch-by-batch" in name:
        if "generator" in name:
            return "Generator Loss"
        elif "discriminator" in name:
            return "Discriminator Loss"
    elif "semantic_loss" in name:
        if "timed" in name:
            return "Timed Semantic Loss"
        else:
            return "Semantic Loss"
    elif "fuzzy_logic" in name:
        if "timed" in name:
            return "Times fuzzy logic based loss"
        else:
            return "Fuzzy logic based loss"
    elif "monsters_tile_number" in name:
        return "Average monsters tile number per sample"
    elif "cannons_number" in name:
        return "Cannons tiles per sample"
    elif "pipes_number" in name:
        return "Pipes tiles per sample"
    elif "first_to_second_distance" in name:
        return "First to second logits difference"
    elif "accuracy" in name:
        return "Accuracy"
    elif "f1" in name:
        return "F1-score"
    elif "precision" in name:
        return "Precision"
    elif "recall" in name:
        return "Recall"
    elif "test_loss" in name:
        return "Test Loss"
    elif "training_loss" in name:
        return "Training Loss"
    elif "test_kl" in name:
        return "Kullback–Leibler Divergence"
    elif "test_mean_squared_error" in name:
        return "Squared Mean Error"
    else:
        return name


def add_comma(n):
    return n[0] + "." + n[1:]

def exp_translate_name(name):
    """
    Something quickly put togheter to format experiment names.
    :param name: the name of the experiment
    :return: the name to shown on the plot
    """

    if "reachability_astar" in name:
        parts = name.split('-')
        game = parts[0]
        level = "{}-{}".format(parts[1], parts[2])
        if "base" not in name:
            sl_weight = parts[4]
            from_epoch = parts[5]
            name = "can:lev_{}_with_sl_{}_from_{}_constr_2".format(level, sl_weight, from_epoch)
        else:
            name = "gan:lev_{}".format(level)

    elif "reachability" in name:
        parts = name.split('-')
        game = parts[0]
        level = "{}-{}".format(parts[1], parts[2])
        if "base" not in name:
            sl_weight = parts[4]
            from_epoch = parts[5]
            name = "can:lev_{}_with_sl_{}_from_{}_constr_1".format(level, sl_weight, from_epoch)
        else:
            name = "gan:lev_{}".format(level)

    elif "monsters" in name:
        parts = name.split('-')
        game = parts[0]
        level = "{}-{}".format(parts[1], parts[2])
        if "base" not in name:
            if "semantic" in name:
                sl_weight = parts[6]
                from_epoch = parts[7]
                name = "can:lev_{}_with_sl_{}_from_{}".format(level, add_comma(sl_weight), from_epoch)
            else:
                fuzzy_weight = parts[7]
                from_epoch = parts[8]
                if "cnf" in name:
                    name = "can:lev_{}_with_fl_cnf_{}_from_{}".format(level, add_comma(fuzzy_weight), from_epoch)
                elif "dnf" in name:
                    name = "can:lev_{}_with_fl_dnf_{}_from_{}".format(level, add_comma(fuzzy_weight), from_epoch)
                else:
                    name = "can:lev_{}_with_fl_{}_from_{}".format(level, add_comma(fuzzy_weight), from_epoch)
        else:
            name = "gan:lev_{}".format(level)

    elif "cannons" in name:
        parts = name.split('-')
        game = parts[0]
        level = "{}-{}".format(parts[1], parts[2])
        if "base" not in name:
            sl_weight = parts[4]
            from_epoch = parts[5]
            name = "can:lev_{}_with_sl_{}_from_{}".format(level, sl_weight, from_epoch)
        else:
            name = "gan:lev_{}".format(level)

    elif "pipes" in name:
        parts = name.split('-')
        game = parts[0]
        level = "{}-{}".format(parts[1], parts[2])
        if "base" not in name:
            sl_weight = parts[4]
            ploss_weight = parts[5]
            from_epoch = parts[6]
            name = "can:lev_{}_with_sl_{}_npipes_{}_from_{}".format(level, sl_weight, ploss_weight, from_epoch)
        else:
            name = "gan:lev_{}".format(level)

    elif "onehot" in name:
        parts = name.split('-')
        game = parts[0]
        level = "{}-{}".format(parts[1], parts[2])
        if "base" not in name:
            sl_weight = parts[4]
            from_epoch = parts[5]
            name = "can:lev_{}_with_sl_{}_from_{}".format(level, sl_weight, from_epoch)
        else:
            name = "gan:lev_{}".format(level)
    return name 


def correct_for_weights(cols, exp):
    adv = cols[0]
    for c in cols[1:]:
        weight = float(exp.split(" ")[-1]) if "0." in exp else 1.0
        print("Correcting weight for experiment %s, weight: %s" % (exp, weight))
        adv -= weight * c
    cols[0] = adv
    return cols


def plot_column(column_name, exp_names, data, ybot=0.0, ytop=1.05,
                s=None,
                std_runs=False,
                title_fontsize=26,
                legend_fontsize=24,
                tick_fontsize=22,
                image_size=(1920, 1080)):

    dpi = 200
    plt.figure(figsize=(image_size[0] / dpi, image_size[1] / dpi), dpi=dpi)

    for d in data:
        mean = np.mean(d, axis=1) if std_runs else d
        x = np.arange(len(mean))

        if s is not None:
            bspl = splrep(x, mean, s=s)
            mean = splev(x, bspl)

        plt.plot(x, mean)
        #if std_runs:
        #    std = np.std(d, axis=1)
        #    plt.fill_between(x, mean - std, mean + std, alpha=0.1, linewidth=0)

    title = col_translate_name(column_name).split('_')[0]# + ("_smoothed" if s is not None else "")
    plt.title(title, fontdict={"fontsize": title_fontsize})
    plt.legend(exp_names, fontsize=legend_fontsize, markerscale=40.0)
    plt.ylim(bottom=ybot, top=ytop)

    # set fontsize of labels on the axis
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    ###### grid
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    # Show the minor grid lines with light grey lines
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.savefig(title + ".png", bbox_inches="tight")
    plt.clf()


def compare_columns(c1_name, c2_name, exp_names, c1_data, c2_data, ybot=0.0, ytop=1.05, s=None):
    x = np.arange(len(c1_data[0]))
    dpi = 96
    plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)
    c1_name = col_translate_name(c1_name)
    c2_name = col_translate_name(c2_name)

    plot_names = []
    for d1, d2, exp_name in zip(c1_data, c2_data, exp_names):
        if s is None:
            plt.plot(x, d1)
            plt.plot(x, d2)
        else:
            bspl = splrep(x, d1, s=s)
            bspl_y = splev(x, bspl)
            plt.plot(x, bspl_y)
            bspl = splrep(x, d2, s=s)
            bspl_y = splev(x, bspl)
            plt.plot(x, bspl_y)
        plot_names.append(exp_name + " --- " + c1_name)
        plot_names.append(exp_name + " --- " + c2_name)

    title = c1_name + ", " + c2_name
    title = title + (", smoothed" if s is not None else "")
    plt.title(title, fontdict={"fontsize": 20})
    plt.legend(plot_names, fontsize=20, markerscale=40.0)
    plt.ylim(bottom=ybot, top=ytop)

    ###### grid
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    title = title.replace(", ", "-")
    plt.savefig(title + ".png")
    plt.clf()


def get_experiment_events(paths):
    # find tensorboard event files
    if isinstance(paths, str):
        paths = [paths]

    event_files = []
    exp_names = []

    for path in paths:
        for root, _, files in os.walk(path):
            for f in files:
                if f.startswith("events.out"):
                    fullpath = os.path.join(root, f)
                    event_files.append(fullpath)
                    exp_names.append(os.path.dirname(os.path.normpath(fullpath)).split("/")[-1][:-7])

    return exp_names, event_files


def get_columns_from_event(exp_name, event_file, columns):
    """
    Get the mean of each provided column
    :param exp_name:
    :param event_file:
    :param columns:
    :return:
    """
    print("Opening %s" % exp_name)
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                                       event_accumulator.IMAGES: 4,
                                                                       event_accumulator.AUDIO: 4,
                                                                       event_accumulator.SCALARS: 0,
                                                                       event_accumulator.HISTOGRAMS: 1, })
    returned_data = []
    ea.Reload()
    tags = ea.Tags()["scalars"]

    for col in columns:
        if col in tags:
            data = pd.DataFrame(ea.Scalars(col))
            data = data[["step", "value"]]
            data = data.groupby("step").mean()
            data = data.values
            data = np.squeeze(data, axis=1)
        else:
            raise AssertionError("col {} not found, returning all zeros, available columns {}".format(col, tags))
        returned_data.append(data)
    return returned_data


def get_exp_name_with_no_run(name):
    if "run" in name:
        name = "_".join(name.split("_")[:-1])
    return name
