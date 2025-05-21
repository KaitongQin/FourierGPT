import numpy as np
import pandas as pd
import pywt
import tqdm
from scipy import signal
from typing import Union
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='', help='input file')
parser.add_argument('--output', '-o', type=str, default='', help='output file or dir')
parser.add_argument('--type', type=str, default='dwt', choices=['dwt', 'cwt'])
parser.add_argument('--preprocess', '-p', type=str, default='none', choices=['none', 'zscore', 'minmax', 'log', 'logzs'])
parser.add_argument('--value', type=str, default='norm', choices=['norm', 'real', 'imag'])
parser.add_argument('--require_sid', action='store_true', default=True, help='if true, append sequence id to output file')
parser.add_argument('--verbose', action='store_true', help='verbose mode')


class WaveletProcessor(object):
    def __init__(self, wavelet_type='dwt', wavelet_name='db4', preprocess='none', require_sid=True, verbose=False):
        self.wavelet_type = wavelet_type  # 'dwt' or 'cwt'
        self.wavelet_name = wavelet_name
        self.preprocess = preprocess
        self.require_sid = require_sid
        self.verbose = verbose

    def _read_data(self, data_file: str, N: int = np.inf):
        data = []
        with open(data_file, 'r') as f:
            count = 0
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                num = list(map(float, line.split()))
                data.append(num)
                count += 1
                if count >= N:
                    break
        return data

    def _preprocess(self, input_data: list):
        data = input_data.copy()
        eps = 1e-6
        if self.preprocess == 'zscore':
            return [(np.array(d) - np.mean(d)) / (np.std(d) + eps) for d in data]
        elif self.preprocess == 'minmax':
            return [(np.array(d) - np.min(d)) / (np.max(d) - np.min(d) + eps) for d in data]
        elif self.preprocess == 'log':
            return [np.log(np.array(d) + 1) for d in data]
        elif self.preprocess == 'logzs':
            return [(np.log(np.array(d) + 1) - np.mean(np.log(np.array(d) + 1))) / (np.std(np.log(np.array(d) + 1)) + eps) for d in data]
        elif self.preprocess != 'none':
            raise ValueError(f'Unknown preprocess method: {self.preprocess}')
        return data

    def _dwt(self, data: np.ndarray):
        coeffs = pywt.wavedec(data, self.wavelet_name)
        freqs, powers, levels = [], [], []
        for level, coef in enumerate(coeffs):
            freqs.extend([level] * len(coef))  # pseudo-frequency by level
            powers.extend(np.abs(coef))
            levels.extend(np.arange(len(coef)))  # time-like index
        return freqs, powers, levels

    def _cwt(self, data: np.ndarray):
        scales = np.arange(1, 32)
        coef, freqs = pywt.cwt(data, scales, 'morl')
        scales_len, time_len = coef.shape

        freq_list, power_list, time_idx = [], [], []
        for i in range(scales_len):
            for j in range(time_len):
                freq_list.append(scales[i])
                power_list.append(np.abs(coef[i, j]))
                time_idx.append(j)
        return freq_list, power_list, time_idx

    def _batch_transform(self, data: list[np.ndarray]):
        all_freqs, all_powers, all_sids, all_tidx = [], [], [], []
        for i, d in tqdm.tqdm(enumerate(data), total=len(data), disable=not self.verbose):
            try:
                if self.wavelet_type == 'dwt':
                    freqs, powers, tidx = self._dwt(d)
                elif self.wavelet_type == 'cwt':
                    freqs, powers, tidx = self._cwt(d)
                else:
                    raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")
            except Exception as e:
                print(f"Error in sample {i}: {e}")
                continue
            all_freqs.extend(freqs)
            all_powers.extend(powers)
            all_tidx.extend(tidx)
            if self.require_sid:
                all_sids.extend([i] * len(freqs))
        return all_freqs, all_powers, all_sids, all_tidx

    def process(self, input_data: Union[str, list]):
        if isinstance(input_data, str):
            raw_data = self._read_data(input_data)
            data = [np.asarray(d) for d in raw_data]
        else:
            data = [np.asarray(d) for d in input_data]

        data = self._preprocess(data)

        freqs, powers, sids, tidx = self._batch_transform(data)

        if self.require_sid:
            df = pd.DataFrame({
                'sid': sids,
                'freq': freqs,
                'power': powers
            })
        else:
            df = pd.DataFrame({
                'freq': freqs,
                'power': powers
            })
        return df

def main(args):
    wavelet_processor = WaveletProcessor(wavelet_type=args.type, 
                                 preprocess=args.preprocess,
                                 require_sid=args.require_sid)
    df = wavelet_processor.process(args.input)
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    datasets = ['pubmed', 'writing', 'xsum']
    models = ['gpt-3.5', 'gpt-3', 'gpt-4']
    labels = ['original', 'sampled']
    generated_models = ['bigram', 'gpt2xl', 'mistral']
    for dataset in datasets:
        for model in models:
            for label in labels:
                for generated_model in generated_models:
                    input_path = f"data/{dataset}/{dataset}_{model}.{label}.{generated_model}.nll.txt"
                    output_path = f"data/{dataset}/{dataset}_{model}.{label}.{generated_model}.nllzs.wavelet_{args.type}.txt"
                    args.input = input_path
                    args.output = output_path
                    main(args)
    # args.input = "data/pubmed/pubmed_gpt-4.sampled.mistral.nll.txt"
    # main(args)


# python run_wavelet.py --input "data/{dataset}/{dataset}_{model}.{label}.{generated}.nll.txt" --output "data/{dataset}/{dataset}_{model}.{label}.{generated}.nllzs.wavelet.txt" --type dwt --preprocess zscore --require_sid