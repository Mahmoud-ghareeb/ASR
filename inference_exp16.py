
import logging

logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

import nemo
import nemo.collections.asr as nemo_asr
import Levenshtein

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth',None)

import os
import glob
from pathlib import Path

from filehash import FileHash
sha256hasher = FileHash('sha256')

import sys
import time
from datetime import datetime


import torch
device = torch.device("cpu")

import itertools

def calcWer(s1, s2):
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    return Levenshtein.distance(''.join(w1), ''.join(w2)) / len(''.join(w2))

def eval_transducer(model,audio_filepath, target):
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=model)
    asr_model.eval()
    transcription = asr_model.transcribe([audio_filepath])[0][0]
    sample_wer = calcWer(transcription, target)
    return [transcription,sample_wer]

def time_diff(ckpt_ext, nemo_ext):
    m_time1 = os.path.getmtime(ckpt_ext)
    m_time2 = os.path.getmtime(nemo_ext)
    result = -100 < m_time1 - m_time2 < 0
    return result

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=None,
        required=True,
        help="Path to nemo checkpoints saved during training. Ex: /raid/nemo_experiments/multimodal/checkpoints",

    )
    parser.add_argument(
        "--sets",
        type=str,
        default=None,
        required=True,
        help="choose set that you testing"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    folder_datename = args.checkpoint_folder.split("/")[-3]
    folder_projectname = args.checkpoint_folder.split("/")[-4]
    set_name = args.sets.split("/")[-1][:-5]
    checkpoints_path = f"{args.checkpoint_folder}/*" 
    checkpoint_list = []

    ckpt_data_path = f"{args.checkpoint_folder}/*.ckpt"
    ckpt_data_list = sorted(glob.glob(ckpt_data_path), key=os.path.getmtime)
    nemo_data_path = f"{args.checkpoint_folder}/*.nemo"
    nemo_data_list = sorted(glob.glob(nemo_data_path), key=os.path.getmtime)

    while len(ckpt_data_list):
        ckpt = ckpt_data_list.pop(0)
        for nemo_ckpt in nemo_data_list:
            if(time_diff(ckpt, nemo_ckpt)):
                checkpoint_list.append(nemo_ckpt)

    print(len(checkpoint_list))

    for ll in range(len(checkpoint_list) - 1):
        if (sha256hasher.hash_file(checkpoint_list[ll]) == sha256hasher.hash_file(checkpoint_list[ll + 1])):
            checkpoint_list.remove(checkpoint_list[ll])
    print(len(checkpoint_list))


    df_path = f"{args.sets}"
    df = pd.read_json(df_path, lines=True)

    samples_length = len(df)

    wer_dict = {}
    result_list = []
    cnt = 1 

    for name_model in checkpoint_list:
        print(cnt)
        data_list = []
        df1 = pd.DataFrame()
        df1[["inference", "wer"]] = df.apply(lambda x: eval_transducer(name_model, x.audio_filepath, x.golden_transcription), axis=1,result_type='expand')
        df1["golden"] = df["golden_transcription"]
        samples_wer = df1["wer"].sum()
        avg_wer = samples_wer/samples_length

        wer_dict["wer"] = [samples_wer/samples_length]
        wer_dict["inference"] = [os.path.basename(name_model)[:-5]]
        wer_dict["golden"] = ["golden"]

        df1 = pd.concat([df1, pd.DataFrame(wer_dict)], ignore_index=True)
        cols = df1.columns.tolist()
        cols = cols[-1:] + cols[:-1] 
        df1 = df1[cols]
        data_list = list(itertools.chain.from_iterable(df1.values.tolist()))

        result_list.append(data_list)

        cnt+=1

    arr = np.array(result_list).T

    df = pd.concat([df, pd.DataFrame({"audio_filepath": ["avg_wer"]})], ignore_index=True)
    index = pd.MultiIndex.from_product([df["audio_filepath"], ["golden","inference", "wer"]], names=["audio", "results"])
    columns = pd.MultiIndex.from_product([[df_path], map(lambda x: os.path.basename(x)[:-5] , checkpoint_list)], names=["set", "model"])
    data = pd.DataFrame(arr, index=index, columns=columns)

    with open(f'./result_reports/{folder_projectname}_{folder_datename}_{set_name}.csv', 'w', encoding='utf-8') as file:
        data.to_csv(file)

logging.disable(logging.NOTSET)
