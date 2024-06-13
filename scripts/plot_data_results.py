import os
import click
import numpy as np
import uproot as ur

from plot_test_results import plot_results

def extract_results(input_path, bins, knots, ndf):
    """
    Extract results from TRUEE written root files.
    This requires TRUEE to be run in test mode!

    Parameters
    ----------
    input_path: str
        Input file.

    key: str
        Result key. The key changes with TRUEE settings and 
        thus needs to be defined individually.

    Returns
    -------
    results: numpy.array
        Unfolding result array.

    errors: numpy.array
        Unfolding result errors.

    mc_truth: numpy.array
        True Monte Carlo distribution.

    bins: numpy.array
        Bins.
    """
    key_data = f"events_result_bins_{bins}_knots_{knots}_degFree_{ndf}"
    with ur.open(input_path) as f:
        results, bins = f["RealDataResults"][key_data].to_numpy()
        errors = np.array(f["RealDataResults"][key_data].errors())

    return results, errors, bins

def get_data(file, key):
    with ur.open(file) as f:
        data = np.array(f["tree"][key].array())

    return data
    
@click.command()
@click.argument("input_path")
@click.argument("input_mc")
@click.argument("bins")
@click.argument("knots")
@click.argument("ndf")
@click.argument("mc_truth_key")
@click.argument("output_path")
def main(input_path, input_mc, bins, knots, ndf, mc_truth_key, output_path):
    # Extract results from file
    results, errors, binning = extract_results(
        input_path,
        bins,
        knots,
        ndf,
    )

    bincenters = binning[:-1] + np.diff(binning)/2

    mc_data = get_data(input_mc, mc_truth_key)
    mc_truth = np.histogram(mc_data, 10**binning)[0]

    # Plot Results
    fig, axes1, axes2 = plot_results(results, errors, mc_truth, bincenters)

    # Save Plot
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    key = f"data_result_{bins}bins_{knots}knots_{ndf}ndf"
    fig.savefig(os.path.join(output_path, f"{key}.pdf"))

if __name__=="__main__":
    main()


    
