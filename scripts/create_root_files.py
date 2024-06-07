import os
import click
import numpy as np
import pandas as pd
import uproot as ur
import yaml

from functools import reduce

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

    # Write root file
    if not os.path.exists(cfg["output_path"]):
        os.mkdir(cfg["output_path"])

    with ur.recreate(os.path.join(cfg["output_path"], cfg["output_file"])) as f:
        f["tree"] = df_mc

if __name__=="__main__":
    main()