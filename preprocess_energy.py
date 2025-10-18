#!/usr/bin/env python3
import json
import os
import numpy.typing
import argparse
from gen_benchmark import get_native_paths, load_native_trajs_stride, machines
from report_generator.tica_plots import calc_atom_distance, TicaModel, generate_tica_model_from_scratch
from report_generator.gpu_kde import gaussian_kde_gpu
from report_generator.cache_loading import load_cache_or_make_new
import numpy as np
import pathlib
from tqdm import tqdm

k_B = 0.001987204259 #kilocalories/mole/Kelvin from here https://en.wikipedia.org/wiki/Boltzmann_constant#Value_in_different_units
T = 300

prior_params = json.load(open("/media/DATA_18_TB_1/andy/models/benchmark_trained_trp-cage_higher_learning_rate/result-2024.11.06-18.50.45/prior_params.json", "r"))    

machine = "bizon"

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--protein-name", required=True, help="name of the protein")
    arg_parser.add_argument("--output-path", required=True, help="output of preprocess.py to add energies to")
    arg_parser.add_argument("--num-components", default=None, type=int, help="output of preprocess.py to add energies to")
    arg_parser.add_argument("--cache-load", action=argparse.BooleanOptionalAction, default=False, help="output of preprocess.py to add energies to")
    arg_parser.add_argument("--kde-strat", type=str, default="gaussian", choices={"gaussian", "histogram"}, help="which kde to use to calculate the pdf")

    args = arg_parser.parse_args()

    input_path = os.path.join(machines[machine].data_300_path, args.protein_name)

    paths = get_native_paths(input_path, False)
    native_trajs, _ = load_native_trajs_stride(paths, prior_params, 1, machines[machine].cache_path, args.protein_name, False, T)
    
    if args.cache_load:
        tica_pdfs = np.load("ticas_x.npy")
    else:
        NATIVE_PATHS_STRIDE = 1
        component_analysis_cache_filename = os.path.join(machines[machine].cache_path, f"{args.protein_name}_{T}K_stride{NATIVE_PATHS_STRIDE}.tica")
        model: TicaModel = load_cache_or_make_new(
            pathlib.Path(component_analysis_cache_filename),
            lambda: generate_tica_model_from_scratch(
                paths,
                prior_params,
                NATIVE_PATHS_STRIDE),
            TicaModel,
            False
        )


        atom_distances: list[numpy.typing.NDArray] = [calc_atom_distance(traj.trajectory) for traj in tqdm(native_trajs, desc="calculating atom distances")]
        tica_datas: list[numpy.typing.NDArray] = model.decompose(atom_distances)

        
        tica_datas_all: numpy.typing.NDArray = numpy.concatenate(tica_datas)

        print("Calculating PDFs")
        num_tica_components: int = args.num_components if args.num_components is not None else tica_datas_all.shape[1]
        tica_pdfs: list[list[numpy.typing.NDArray]] = []
        for x in range(num_tica_components):
            print(f"Calculating pdf of component {x}/{num_tica_components}")
            match args.kde_strat:
                case "gaussian":
                    tica_pdfs.append(calc_pdf_gaussian(tica_datas_all, tica_datas, x))
                case "histogram":
                    tica_pdfs.append(calc_pdf_histogram(tica_datas_all, tica_datas, x))
                case _:
                    assert False
                    
        np.save("ticas_x.npy", tica_pdfs)
    
    energies = -k_B * T * np.sum(np.nan_to_num(np.log(tica_pdfs)), axis=0)

    for i, native_traj in enumerate(native_trajs):
        pdb_id = os.path.split(os.path.split(os.path.split(native_traj.path)[0])[0])[1]
        prior_energies = numpy.load(os.path.join(args.output_path, pdb_id, "raw", "prior_energy.npy")).flatten()
        assert energies[i, :].shape == prior_energies.shape
        delta_energies = energies[i, :] - prior_energies
        path = os.path.join(args.output_path, pdb_id, "raw", "tica_delta_energies.npy")
        np.save(path, delta_energies)


def calc_pdf_gaussian(all_datas_concated: numpy.typing.NDArray, tica_datas: list[numpy.typing.NDArray], component: int) -> list[numpy.typing.NDArray]:
    component_tica = np.array([all_datas_concated[::500, component]])
    output: list[numpy.typing.NDArray] = []
    for data in tqdm(tica_datas):
        gpu_data = gaussian_kde_gpu(component_tica.T, np.array([data[::1, component]]).T)
        output.append(gpu_data)
    return output

def calc_pdf_histogram(all_datas_concated: numpy.typing.NDArray, tica_datas: list[numpy.typing.NDArray], component: int) -> list[numpy.typing.NDArray]:
    num_datas = all_datas_concated.shape[0]
    num_buckets = int(num_datas / 1000)
    all_data_tica = all_datas_concated[:, component]
    component_min, component_max = numpy.min(all_data_tica), numpy.max(all_data_tica)

    component_bucket = numpy.astype(((all_data_tica - component_min) / (component_max - component_min)) * num_buckets, int)
    assert (component_bucket >= 0.0).all() and (component_bucket <= num_buckets).all()
    bucket_count = numpy.zeros(num_buckets + 1, dtype=int)
    for bucket in component_bucket:
        bucket_count[bucket] += 1

    bucket_probabilities = bucket_count / float(num_datas)
    assert(abs(numpy.sum(bucket_probabilities) - 1.0) < 0.0001)

    output: list[numpy.typing.NDArray] = []
    for data in tqdm(tica_datas):
        data_bucket = numpy.astype(((data[:, component] - component_min) / (component_max - component_min)) * num_buckets, int)
        assert (data_bucket >= 0.0).all() and (data_bucket <= num_buckets).all()
        tica_pdf = numpy.array([bucket_probabilities[bucket] for bucket in data_bucket])
        output.append(tica_pdf)
    return output
        


if __name__ == "__main__":
    main()
