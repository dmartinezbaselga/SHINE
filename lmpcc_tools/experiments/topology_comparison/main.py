#!/usr/bin/env python3

import sys, os
sys.path.append("..")

import warnings

from scripts.experiments.topology_comparison.set_parameters import set_parameters
from scripts import set_parameters as parameters

from scripts import helpers, compare

simulations = [
    "random-2_straight",
    "random-4_straight",
    "random-6_straight"
]

def table_header(table, f, metrics, simulation, full_caption):
    f.write(
        '\\begin{table}[t]\n'
        '\\centering\n'
        )
    f.write(
        '\\caption{Quantitative measurement of topology comparison functions for scenarios with 2, 4 and 6 pedestrians.}\n'
    )

    f.write('\\begin{tabular}{|l|')
    for i in range(len(table.headers)):
        if i < len(table.headers) - 1:
            f.write(f'{table.alignments[i]}' + "|")
        else:
            f.write(f'{table.alignments[i]}' + "|}")
    f.write("\n")

    f.write("\\hline\\textbf{Scenario} & ")
    for i, header in enumerate(table.headers):
        f.write('\\textbf{' + header + "}")
        table.write_next(f, i, True)

def table_footer(table, f):
    f.write(
        "\\end{tabular}\n"
        "\\label{tab:topology_comparison}\n"
        "\\end{table}\n"
        )

def process():
    table()

def table(simulation="random-4_straight"):
    table_folder, _ = helpers.get_latex_folder_name()

    metrics = []
    for simulation in simulations:
        metrics.append(compare.load_table_data("UVD-" + simulation))
        metrics[-1][0]["name"] = "UVD"

        metrics[-1].append(compare.load_table_data("Homology-" + simulation)[0])
        metrics[-1][1]["name"] = "Homology"

    os.makedirs(table_folder, exist_ok=True)
    table = compare.CombinedLatexTable(table_folder + "results" + ".tex", simulations)
    table.set_header(table_header)
    table.set_footer(table_footer)

    table.add_data(
        "Method", lambda metrics, highlight: metrics["name"], align="l"
        )

    table.add_data(
        "Paths Avg (std) [\\#]", lambda metrics, highlight: f'{highlight(metrics["num paths"]["mean"], 2)} ({metrics["num paths"]["std"]:.1f})'
        )

    table.add_data(
        "Runtime [ms]", lambda metrics, highlight: f'{highlight(metrics["runtime"]["mean"] * 1000., 1)} ({metrics["runtime"]["std"] * 1000.:.1f})', align="l"
        )

    table.write_table(metrics, simulation, False)

    print(helpers.bcolors.HEADER + "----------------------" + helpers.bcolors.ENDC)
    print(helpers.bcolors.OKGREEN + "Done!" + helpers.bcolors.ENDC + " Saved table in " + table.filename)


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == "parameters":
        set_parameters(sys.argv[2])
    elif stage == "restore":
        parameters.remove_config()
    elif stage == "process":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            process()

