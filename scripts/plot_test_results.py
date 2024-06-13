import os
import click
import numpy as np
import uproot as ur
import matplotlib.pyplot as plt

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

    bincenters: numpy.array
        Bin centers.
    """
    key_test = f"result_{bins}bins_{knots}knots_{ndf}ndf"
    key_data = f"events_result_bins_{bins}_knots_{knots}_degFree_{ndf}"
    with ur.open(input_path) as f:
        results = np.array(f["RealDataResults"][key_data].values())
        errors = np.array(f["RealDataResults"][key_data].errors())
        mc_truth = np.array(f["TestResults"][key_test]["ndiff_mc_truth"].array())
        bincenters = np.array(f["TestResults"][key_test]["bin_center"].array())

    return results, errors, mc_truth, bincenters

def plot_results(results, errors, mc_truth, bincenters):
    fig, (axes1, axes2) = plt.subplots(
        2,
        1,
        figsize=(7.5, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    # Axes 1
    axes1.errorbar(
        bincenters,
        mc_truth,
        yerr=np.sqrt(mc_truth),
        color="k",
        marker="s",
        markersize=4,
        ls="",
        capsize=4,
        elinewidth=0,
        label="MC",
    )
    axes1.errorbar(
        bincenters,
        results,
        yerr=errors,
        color="r",
        marker=".",
        ls="",
        capsize=4,
        elinewidth=0,
        label="Unfolded",
    )
    axes1.set_ylabel("Counts")
    axes1.set_yscale("log")
    axes1.tick_params(bottom=False)
    axes1.legend(loc="best", fontsize=10, ncol=2)

    # Axes 2
    axes2.axhline(1, color="k", lw=0.5)
    axes2.errorbar(
        bincenters,
        results/mc_truth,
        yerr=errors/mc_truth,
        color="r",
        marker=".",
        ls="",
        capsize=4,
        elinewidth=0,
    )
    axes2.set_xlabel("Propagation length / m.w.e.")
    axes2.set_ylabel(
        r"$\frac{f_{\mathrm{unf}}}{f_{\mathrm{MC}}}$", fontsize=20
    )
    limit = np.abs(1 - (results/mc_truth)).max() * 1.2
    axes2.set_ylim(1 - limit, 1 + limit)

    return fig, axes1, axes2

    
@click.command()
@click.argument("input_path")
@click.argument("bins")
@click.argument("knots")
@click.argument("ndf")
@click.argument("output_path")
def main(input_path, bins, knots, ndf, output_path):
    # Extract results from file
    results, errors, mc_truth, bincenters = extract_results(
        input_path,
        bins,
        knots,
        ndf,
    )

    # Plot Results
    fig, axes1, axes2 = plot_results(results, errors, mc_truth, bincenters)

    # Save Plot
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    key = f"result_{bins}bins_{knots}knots_{ndf}ndf"
    fig.savefig(os.path.join(output_path, f"{key}.pdf"))

if __name__=="__main__":
    main()


    
