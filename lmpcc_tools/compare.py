import argparse
import sys, os
import json
import numpy as np

sys.path.append("..")

import scripts.helpers as helpers
from scripts.helpers import print_value, bcolors, PROJECT_FOLDER


def highlight(value, decimals):
    value = f'{value:.{decimals}f}'
    return "\\textbf{" + str(value) + "}"


def no_highlight(value, decimals):
    return f'{value:.{decimals}f}'


def find_highlighted_value(text):
    highlight_split = text.split("\\textbf{")
    if len(highlight_split) == 1:
        return -1
    else:
        value = float(highlight_split[1].split("}")[0])
        return value


def include_method(text, method, metrics):
    result = text(metrics) if metrics['name'] == method else "-"
    return result


def exclude_method(text, method, metrics):
    result = text if metrics['name'] != method else "-"
    return result


def clean_simulation_name(simulation):
    if len(simulation.split("-")) > 2:
        prepend = simulation.split("-")[0]
        if prepend == "fully":
            simulation = simulation.split("-")[1] + "-" + simulation.split("-")[2]
        else:
            risk = simulation.split("-")[0].split(" ")[0]
            if risk == "1":
                return "High Risk"
            elif risk == "01":
                return "Medium Risk"
            elif risk == "001":
                return "Low Risk"
    elif len(simulation.split("-")) == 1:
        simulation = simulation.split("_")[0]
        return simulation

    sim_type = simulation.split("-")[0]
    sim_type = sim_type.replace("_fast", "")
    sim_type = sim_type.replace("_social", " Social")
    sim_peds = simulation.split("-")[1].split("_")[0]

    # return "" + sim_type.title() + " " + sim_peds
    return sim_peds


def clean_method_name(name):
    if name == "GMPCCNO":
        name = "GMPCC (no original)"
    elif name == "frenet-planner":
        name = "Fren\\'et-Planner"
    elif name == "Guidance-MPCCNO":
        name = "Guidance-MPCC~[21]"
    elif name == "LMPCC":
        name = "LMPCC~[4]"
    return name


def caption_lookup(name):
    if name == "results":
        return lambda num_experiments, num_scenarios: f"\\caption{{Comparison between GMPCC and baselines" \
                                                      f" in {num_scenarios} scenarios of varying risk, where each experiment is " \
                                                      f"repeated {num_experiments} times. Metrics are displayed as ``average (std)'' except for collisions" \
                                                      f" that are separated based on their " \
                                                      f"velocity in active ($v > 0.1$ m/s) and passive velocities. Infeasible denotes the " \
                                                      f"percentage of experiments where the planner became infeasible or ran out of time to find a " \
                                                      f"trajectory at least once / twice / thrice.}}"
    elif name == "uncertainty_results":
        return lambda num_experiments, num_scenarios: f"\\caption{{Comparison between CC-MPC and CC-MPC in the GMPCC framework" \
                                                      f" in {num_scenarios} scenarios of varying risk, where each experiment is " \
                                                      f"repeated {num_experiments} times. Results are displayed as in Tab.~\\ref{{tab:results}}.}}"
    elif name == "social_results":
        return lambda num_experiments, num_scenarios: f"\\caption{{Comparison between GMPCC and baselines for pedestrians following the social forces " \
                                     f"model~\\cite{{helbing_social_1995}} and motion predictions of the pedestrians assume constant velocity. Metrics are collected in " \
                                     f"{num_scenarios} scenarios, each repeated {num_experiments} times. Results are displayed as in Tab.~\\ref{{tab:results}}.}}"
    elif name == "fully_social_results":
        return lambda num_experiments, num_scenarios: f"\\caption{{Comparison between GMPCC and baselines for pedestrians and pedestrian predictions following the social forces " \
                                     f"model~\\cite{{helbing_social_1995}}. Metrics are collected in " \
                                     f"{num_scenarios} scenarios, each repeated {num_experiments} times. Results are displayed as in Tab.~\\ref{{tab:results}}.}}"
    else:
        return lambda num_experiments, num_scenarios: f"\\caption{{Metrics for {num_experiments} experiments in {num_scenarios} scenarios}}"  # $' + str(len(metrics)) +


class LatexTable:

    def __init__(self, name, filename, output_name):
        self.headers = []
        self.data_lambdas = []
        self.alignments = []
        self.highlight_select = []
        self.output_name = output_name

        self.name = name
        self.filename = filename
        self.set_header()
        self.set_footer()

    def write_next(self, f, i):
        if i < len(self.headers) - 1:
            f.write(" & ")
        else:
            f.write(" \\\\\\hline\n")

    def write_table(self, metrics, simulation, full_caption=False):
        f = open(self.filename, "w")

        self.header_lambda(f, metrics, simulation, full_caption)
        self.write_data(f, metrics)
        self.footer_lambda(f)

        f.close()

    def write_header(self, f, metrics, simulation, full_caption):
        d = metrics[0]

        f.write('\\begin{table*}\n'
                '\\centering\n')
        if full_caption:
            f.write(
                '\\caption{Metrics for ' + str(d["num experiments"]) + ' experiments in a ' + clean_simulation_name(simulation) + ' environment with ' + str(d[
                    "num obstacles"]) + ' pedestrians. ' + 'Collisions are severe when the distance between robot and pedestrian is 0.05m less than specified. '
                                                           'Timeouts are cases where the robot is not progressing (usually because it is outside of the road). '
                                                           'All values are denoted as mean (std) over all experiments unless stated otherwise.}\n')
        else:
            f.write(
                '\\caption{Metrics for ' + str(d["num experiments"]) + ' experiments in a ' + clean_simulation_name(simulation) + ' environment with ' + str(
                    d["num obstacles"]) + ' pedestrians.}\n')

        f.write('\\begin{tabular}{|')
        for i in range(len(self.headers)):
            if i < len(self.headers) - 1:
                f.write(f'{self.alignments[i]}' + "|")
            else:
                f.write(f'{self.alignments[i]}' + "|}")
        f.write("\n")

        f.write("\\hline")
        for i, header in enumerate(self.headers):
            f.write('\\textbf{' + header + "}")
            self.write_next(f, i)

    def set_header(self, header_lambda=None):
        if header_lambda is None:
            self.header_lambda = self.write_header
        else:
            self.header_lambda = lambda f, metrics, simulation, full_caption: header_lambda(self, f, metrics, simulation, full_caption)

    def set_footer(self, footer_lambda=None):
        if footer_lambda is None:
            self.footer_lambda = self.close_table
        else:
            self.footer_lambda = lambda f: footer_lambda(self, f)

    def get_highlighted(self, metrics):
        # First figure out which is best
        values = np.zeros((len(metrics), len(self.data_lambdas)))
        highlighted = []
        for i, data_text in enumerate(self.data_lambdas):
            for m, method_metrics in enumerate(metrics):
                values[m, i] = find_highlighted_value(data_text(method_metrics, highlight))

            select_value = self.highlight_select[i](values[:, i])
            highlighted.append(list(np.where(values[:, i] == select_value)[
                                        0]))  # There may be more than one  # print(highlighted[-1])  # highlighted.append(self.highlight_select[i](values[:, i]))

        return highlighted

    def write_data(self, f, metrics):
        highlighted = self.get_highlighted(metrics)

        for m, method_metrics in enumerate(metrics):

            for i, data_text in enumerate(self.data_lambdas):
                if m in highlighted[i]:
                    f.write(data_text(method_metrics, highlight))
                else:
                    f.write(data_text(method_metrics, no_highlight))

                self.write_next(f, i)

    def close_table(self, f):
        f.write("\\end{tabular}\n"
                "\\label{tab:" + self.output_name + "}\n"
                                                    "\\end{table*}\n")

    def add_data(self, header_name, data_lambda, align="c", highlight_select=min):
        self.headers.append(header_name)
        self.data_lambdas.append(data_lambda)
        self.alignments.append(align)
        self.highlight_select.append(highlight_select)


class CombinedLatexTable(LatexTable):

    def __init__(self, filename, simulation_list, output_name):
        super().__init__(filename.split(".")[0], filename, output_name)
        self.simulation_list = simulation_list

    def write_next(self, f, i, m):
        if i < len(self.headers) - 1:
            f.write(" & ")
        else:
            if m:
                f.write(" \\\\\\hline\n")
            else:
                f.write(" \\\\\n")

    def write_table(self, metrics, simulation, full_caption=False):
        f = open(self.filename, "w")

        self.header_lambda(f, metrics, simulation, full_caption)
        self.write_data(f, metrics)
        self.footer_lambda(f)

        f.close()

    def write_header(self, f, metrics, simulation, full_caption):
        d = metrics[0]
        f.write('\\begin{table*}\n'
                '\\centering\n')

        f.write(caption_lookup(self.output_name)(d[0]["num experiments"], len(metrics)))
        f.write("\n")
        # if full_caption:
        #     f.write(
        #         '\\caption{Metrics for ' + str(d[0]["num experiments"]) + ' experiments in $' + str(len(metrics)) + '$ scenarios. '
        #         + 'Collisions are severe when the distance between robot and pedestrian is 0.05m less than specified. '
        #           'Timeouts are cases where the robot is not progressing (usually because it is outside of the road). '
        #           'All values are denoted as mean (std) over all experiments unless stated otherwise.}\n')
        # else:
        #     f.write('\\caption{Metrics for ' + str(d[0]["num experiments"]) + ' experiments in $' + str(len(metrics)) +
        #             '$ scenarios.}\n')

        f.write('\\begin{tabular}{|l|')
        for i in range(len(self.headers)):
            if i < len(self.headers) - 1:
                f.write(f'{self.alignments[i]}' + "|")
            else:
                f.write(f'{self.alignments[i]}' + "|}")
        f.write("\n")

        f.write("\\hline\\textbf{\#} & ")
        for i, header in enumerate(self.headers):
            f.write('\\textbf{' + header + "}")
            self.write_next(f, i, True)

    def write_data(self, f, metrics):
        for simulation_id, simulation_metrics in enumerate(metrics):  # For all simulations read

            highlighted = self.get_highlighted(simulation_metrics)

            for m, method_metrics in enumerate(simulation_metrics):

                for i, data_text in enumerate(self.data_lambdas):
                    if m == 0 and i == 0:
                        simulation_name = clean_simulation_name(self.simulation_list[simulation_id])
                        f.write("\\multirow{" + str(len(simulation_metrics)) + "}{*}{" + simulation_name + "} & ")
                    elif i == 0:
                        f.write("&")

                    if m in highlighted[i]:
                        f.write(data_text(method_metrics, highlight))
                    else:
                        f.write(data_text(method_metrics, no_highlight))

                    self.write_next(f, i, m == len(simulation_metrics) - 1)


def load_table_data(simulation):
    data_folder, metrics_folder, _ = helpers.get_folder_names()
    experiment_folder = metrics_folder + simulation
    print_value("Simulation", simulation)
    metrics = []
    print(bcolors.OKGREEN + "[Planner Results]" + bcolors.ENDC)
    for filename in os.scandir(experiment_folder):
        if filename.is_file():
            if filename.name.endswith(".json"):
                print_value("Planner", str(filename), tab=True)
                with open(filename) as f:
                    cur_metrics = dict(json.load(f))
                    metrics.append(cur_metrics)
    return metrics


def main(output_name="results.tex", simulations=helpers.SIMULATIONS_IN_TABLE):
    print(bcolors.HEADER + "----- Compare.py ------" + bcolors.ENDC)

    table_folder, _ = helpers.get_latex_folder_name()

    if len(simulations) == 1:
        simulation = simulations[0]
    else:
        simulation = None

    if simulation is None:
        metrics = []
        simulations = sorted(simulations, key=lambda d: clean_simulation_name(d))
        for simulation in simulations:  # helpers.SIMULATIONS_IN_TABLE:
            metrics.append(load_table_data(simulation))

        for i in range(len(metrics)):
            metrics[i] = sorted(metrics[i], key=lambda d: d['name'], reverse=True)

        os.makedirs(table_folder, exist_ok=True)
        table = CombinedLatexTable(table_folder + output_name + ".tex", simulations, output_name)
    else:
        metrics = load_table_data(simulation)

        os.makedirs(table_folder, exist_ok=True)
        table = LatexTable(simulation, table_folder + simulation + ".tex", output_name)

    table.add_data("Method", lambda metrics, highlight: clean_method_name(metrics["name"]), align="l")

    table.add_data("Task Duration [s]",
                   lambda metrics, highlight: f'{highlight(metrics["task duration"]["mean"], 2)} ({metrics["task duration"]["std"]:.2f})')

    # table.add_data("Collisions (Passive / Active)", lambda metrics, highlight: f"{highlight(metrics['num severe collisions'], 0)}" + f" ({metrics['num passive collisions']} / {metrics['num active collisions']})")

    table.add_data("Collisions", lambda metrics, highlight: f"{highlight(metrics['num severe collisions'], 0)}")

    # table.add_data("Infeasible (\\%)", lambda metrics, highlight: exclude_method(
    #     f"{highlight(float(len([i for i in metrics['num infeasible'] if i > 0])) / float(metrics['num experiments']) * 100., 0)} "
    #     f"/ {float(len([i for i in metrics['num infeasible'] if i > 1])) / float(metrics['num experiments']) * 100.:.0f} "
    #     f"/ {float(len([i for i in metrics['num infeasible'] if i > 2])) / float(metrics['num experiments']) * 100.:.0f}", "frenet-planner", metrics))


    # table.add_data("LMPCC (\\%)", lambda metrics, highlight: include_method(lambda metrics: f"{metrics['lmpcc used']:.0f}", "GMPCC", metrics))

    # table.add_data("E", lambda metrics, highlight: f"{metrics['num experiments']}")

    table.add_data("Timeouts",
                   lambda metrics, highlight: f"{highlight(metrics['num timeouts'], 0)}")

    table.add_data("Completed Safely (\\%)",  # Safely
                   lambda metrics, highlight: f'{highlight(round(metrics["success rate"] * 100.), 0)}', highlight_select=np.argmax)

    table.add_data("Runtime [ms]",
                   lambda metrics, highlight: f'{highlight(metrics["runtime"]["mean"] * 1000., 1)} ({metrics["runtime"]["std"] * 1000.:.1f})',
                   align="l")

    table.write_table(metrics, simulation, simulation == "random-4_straight")

    print(bcolors.HEADER + "----------------------" + bcolors.ENDC)
    print(bcolors.OKGREEN + "Done!" + bcolors.ENDC + " Saved table in " + table.filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # By default it will fail with multiple arguments.
    parser.add_argument('--default')

    if len(sys.argv) > 1:
        simulations = parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)
        parser.add_argument('-o', '--output', required=True)

        args = parser.parse_args()
        argsdict = vars(args)
        print(argsdict)

        output_name = sys.argv[1]
        main(argsdict["output"], argsdict["list"])
    else:
        main()
