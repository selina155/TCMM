import os
import csv
import numpy as np

def write_results_to_disk(dataset, metrics):
    # write results to file
    results_dir = os.path.join('results', dataset)
    results_file = os.path.join(results_dir, 'results.csv')
    file_exists = os.path.isfile(results_file)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(results_file, 'a', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows([metrics])
