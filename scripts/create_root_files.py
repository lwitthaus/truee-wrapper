import os
import click
import numpy as np
import pandas as pd
import uproot as ur
import yaml

from functools import reduce
from sklearn.model_selection import train_test_split

def combine_cuts(cuts, data):
    """
    Returns an index mask that combines multiple cuts for
    a given set of data.

    Parameters
    ----------
    cuts : dict = {key1: [cut_value, cut_option], ...}
        Cuts to combine. The keys correspond to the keys in the
        DataFrame that the cuts should be applied to.

        cut_value is the exact value and cut_option has to be in
        [greater_than, less_than, equal_to, unequal_to]

    data : dict / pandas.DataFrame
        Data sample.
    """
    cut_options = ["greater_than", "less_than", "equal_to", "unequal_to"]
    masks = []
    for variable in cuts.keys():
        assert (
            cuts[variable][1] in cut_options
        ), f"Cut option has to be in {cut_options}"

        if cuts[variable][1] == "greater_than":
            masks += [data[variable] > cuts[variable][0]]
        elif cuts[variable][1] == "less_than":
            masks += [data[variable] < cuts[variable][0]]
        elif cuts[variable][1] == "equal_to":
            masks += [data[variable] == cuts[variable][0]]
        else:
            masks += [data[variable] != cuts[variable][0]]

    mask = reduce(np.logical_and, masks)

    return mask

def split_sample(
        df,
        weights=None,
        train_size=0.5,
        n_sample_train=None,
        n_sample_test=None,
        delta_gamma=None,
        shift_key=None,
        seed=42,
    ):
    if weights is None:
        weights = np.ones(len(df))

    df_train, df_test, w_train, w_test = train_test_split(
        df,
        weights,
        train_size=train_size,
        random_state=seed,
    )

    print(f"# train events: {len(df_train)}")
    print(f"# test events: {len(df_test)}")
    
    if n_sample_train is None:
        n_sample_train = len(df_train)

    if n_sample_test is None:
        n_sample_test = len(df_test)

    print(f"Sampling {n_sample_train} train events.")
    print(f"Sampling {n_sample_test} test events.")

    w_train /= np.sum(w_train)
    i_train = np.random.choice(
        a=range(len(df_train)),
        p=w_train,
        size=n_sample_train,
        replace=True,
    )
    df_train = df_train.iloc[i_train]

    if delta_gamma is not None:
        w_test *= df_test.loc[:, shift_key]**delta_gamma

    w_test /= np.sum(w_test)
    i_test = np.random.choice(
        a=range(len(df_test)),
        p=w_test,
        size=n_sample_test,
        replace=True,
    )

    df_test = df_test.iloc[i_test]

    return df_train, df_test

@click.command()
@click.argument("config")
def main(config):
    # Load config file
    with open(config, "r") as stream:
        cfg = yaml.full_load(stream)

    # Read data
    if "is_hdf5" in cfg and cfg["is_hdf5"]:
        raise NotImplementedError("Reading hdf5 files not implemented.")
    else:
        mc = pd.read_pickle(cfg["mc_input_path"])

    # Create output path
    if not os.path.exists(cfg["output_path"]):
        os.mkdir(cfg["output_path"])

    if "seed" not in cfg or cfg["seed"] is None:
        cfg["seed"] = 42

    mask = combine_cuts(cfg["cuts"], mc)
    mc = mc[mask].copy()

    # Isolate obervables and target
    df_mc = mc.loc[:, cfg["columns"]]

    # Check for NaNs
    check_nan = df_mc.isnull().values.any()
    print(f"Any NaNs in DataFrame: {check_nan}")

    if check_nan:
        print("Removing NaNs...")
        df_mc = df_mc.dropna()

    if "check_zero" in cfg:
        for key in cfg["check_zero"]:
            check_zero = (df_mc[key] == 0).any()
            print(f"Key '{key}' contains zeros: {check_zero}")
            if check_zero:
                print("Removing zeros...")
                df_mc = df_mc[df_mc[key] != 0].copy()

    if "sample_split" in cfg and cfg["sample_split"]:
        print("Splitting sample...")   

        if "split_frac" not in cfg or cfg["split_frac"] is None:
            cfg["split_frac"] = 0.5

        if "sample_weight" not in cfg or cfg["sample_weight"] is None:
            weight = None
        else:
            weight = mc.loc[:, cfg["sample_weight"]].to_numpy()

        if "delta_gamma" not in cfg:
            cfg["delta_gamma"] = None
        if cfg["delta_gamma"] is not None:
            assert (
                "shift_key" in cfg and cfg["shift_key"] is not None
            ), "'shift_key' has to be specified if 'delta_gamma' is given."
        if "shift_key" not in cfg:
            cfg["shift_key"] = None

        df_train, df_test = split_sample(
            df=df_mc,
            weights=weight,
            train_size=cfg["split_frac"],
            n_sample_train=cfg["n_sample_train"],
            n_sample_test=cfg["n_sample_test"],
            delta_gamma=cfg["delta_gamma"],
            shift_key=cfg["shift_key"],
            seed=cfg["seed"],
        )
        
        with ur.recreate(os.path.join(cfg["output_path"], "train.root")) as f:
            f["tree"] = df_train

        with ur.recreate(os.path.join(cfg["output_path"], "test.root")) as f:
            f["tree"] = df_test

    else:
        with ur.recreate(os.path.join(cfg["output_path"], cfg["output_file"])) as f:
            f["tree"] = df_mc

if __name__=="__main__":
    main()