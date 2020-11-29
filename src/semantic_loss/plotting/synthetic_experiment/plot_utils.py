import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import splrep, splev
from tensorboard.backend.event_processing import event_accumulator


def exps_translate_color(name):
    if "1.0" in name:
        return "C1"
    elif "0.5" in name:
        return "C7"
    elif "0.25" in name:
        return "C3"
    elif "0.1" in name:
        return "C4"
    elif "0.01" in name:
        return "C5"
    else:
        return "C2"

def col_translate_name(name):
    """
    Something quickly put togheter to format column names.
    :param name:
    :return:
    """
    name = name.split("/")[-1]
    if "Validity" in name:
        return "Validity"
    elif "Novelty" in name:
        return "Novelty"
    elif "Uniqueness" in name:
        return "Uniqueness"
    elif "batch-by-batch" in name:
        return "Adversarial"
    elif "pc_custom_full" in name:
        return "pc_custom_full"
    elif "pc_custom" in name:
        return "pc_custom"
    else:
        return name


def exp_translate_name(name):
    """
    Something quickly put togheter to format experiment names.
    :param name:
    :return:
    """
    to_join = []
    arch = "gan"
    arch = "can" if "can" in name else arch
    arch = "bgan" if "bgan" in name else arch
    arch = "sngan" if "sngan" in name else arch
    arch = "wgan" if "wgan" in name else arch
    arch = "wgan-gp" if "wgan-gp" in name else arch
    to_join.append(arch)
    hinge_loss = "hinge_loss" if "hinge_loss" in name else ""
    to_join.append(hinge_loss)

    hard = "soft_sampling" if "soft_sampling" in name else ""
    to_join.append(hard)

    pc = "parity check" if "pc_no_area" in name else ""
    to_join.append(pc)

    #mix = "Features" if ("mix" in name or ("can" in name and "sem_loss" not in name)) else ""
    #to_join.append(mix)

    itemsets = "Itemsets" if "itemsets" in name else ""
    to_join.append(itemsets)

    semloss = "Semantic Loss" if "sem_loss" in name else ""
    to_join.append(semloss)

    weight = ""
    if "_w" in name:
        weight = [split for split in name.split("_") if "w" in split][0]
        weight = str(weight[1:])
        if weight.startswith("0"):
            weight = "0." + weight[1:]
        else:
            weight = weight + ".0"
        weight = f" $\lambda$ = " + weight
    to_join.append(weight)

    incremental = " * (epoch / max_epochs)" if "incremental" in name else ""
    to_join.append(incremental)

    if "from" in name:
        # get from which epoch we are starting
        # expects stuff like *_from<digit>_* (no underscore between from and epoch number")
        timed = name.find("from") + len("from")
        end_index = name[timed:].find("_")
        if end_index != -1:
            end_index = timed + end_index
            timed = name[timed: end_index]
        else:
            timed = name[timed:]
        timed = "from epoch %s" % timed
        to_join.append(timed)

    to_join = [item for item in to_join if item != ""]
    name = " ".join(to_join)
    return name


def correct_for_weights(cols, exp):
    adv = cols[0]
    for c in cols[1:]:
        weight = float(exp.split(" ")[-1]) if "0." in exp else 1.0
        print("Correcting weight for experiment %s, weight: %s" % (exp, weight))
        adv -= weight * c
    cols[0] = adv
    return cols


def plot_column(column_name, exp_names, data, ybot=0.6, ytop=1.05, s=None, std_runs=False):
    dpi = 96
    plt.figure(figsize=(9,2.5), dpi=70)

    for d, exp_name in zip(data, exp_names):
        mean = np.mean(d, axis=1) if std_runs else d
        x = np.arange(len(mean))

        if s is not None:
            bspl = splrep(x, mean, s=s)
            mean = splev(x, bspl)

        plt.plot(x, mean, linewidth=2, label=exp_name, color=exps_translate_color(exp_name))
        if std_runs:
            std = np.std(d, axis=1)
            plt.fill_between(x, mean - std, mean + std, alpha=0.1, linewidth=0, color=exps_translate_color(exp_name))

    title = col_translate_name(column_name) + ("_smoothed" if s is not None else "")
    #plt.title(title, fontdict={"fontsize": 20})
    plt.xlabel("Training epoch", fontdict={"fontsize": 12})
    plt.ylabel(f"{title} score", fontdict={"fontsize": 12})
    handles, labels = plt.gca().get_legend_handles_labels()
    indices_gan = [i for i, s in enumerate(labels) if 'gan' in s]
    handles_can = handles[:(indices_gan[0])] + handles[indices_gan[0]+1:] + [handles[indices_gan[0]]]
    labels_can = labels[:(indices_gan[0])] + labels[indices_gan[0]+1:] + [labels[indices_gan[0]]]
    labels_gan = len(labels_can) -1
    sorted_indexes = sorted(range(len(labels_can[:-1])), key = lambda k: labels_can[:-1][k], reverse=True)
    sorted_indexes.append(labels_gan)
    temp_exp_names_can = list(filter(lambda elem : "gan" not in elem, exp_names))
    temp_exp_names_gan = list(filter(lambda elem : "can" not in elem, exp_names))
    sorted_exp_names = sorted(temp_exp_names_can, reverse=True) + sorted(temp_exp_names_gan, reverse=True)
    plt.legend([handles_can[idx] for idx in sorted_indexes],
               [labels_can[idx] for idx in sorted_indexes],
               fontsize=10, markerscale=20.0, ncol=2)
    plt.ylim(bottom=ybot, top=ytop)

    # set fontsize of labels on the axis
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=26)

    ###### grid
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
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


def get_experiment_events(path="tensorboard"):
    # find tensorboard event files
    event_files = []
    exp_names = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("events.out"):
                fullpath = os.path.join(root, file)
                event_files.append(fullpath)
                exp_names.append(os.path.dirname(os.path.normpath(fullpath)).split("/")[-1])
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
            print("col %s not found, returning all zeros" % col)
            data = np.zeros((250,))
        returned_data.append(data)
    return returned_data


def get_exp_name_with_no_run(name):
    if "run" in name:
        name = "_".join(name.split("_")[:-1])
    return name
