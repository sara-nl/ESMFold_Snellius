#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script uses Meta's ESMFold for protein structure prediction.
See: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb#scrollTo=90ee986d
"""

# Import necessary libraries
import os
import argparse

from typing import Tuple
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

# TODO: modify for multimers!

def parse_fasta(filename: str):
    sequences = {}
    current_header = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                current_header = line[1:]
                sequences[current_header] = []
            else:
                sequences[current_header].append(line)

    for header in sequences:
        sequences[header] = ''.join(sequences[header])

    return sequences

def parse_fasta_folder(folder_path: str):
    all_fastas = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            full_path = os.path.join(folder_path, filename)
            all_fastas[filename] = parse_fasta(full_path)
    return all_fastas

def get_model(args) -> Tuple[EsmForProteinFolding, AutoTokenizer]:
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    model = EsmForProteinFolding.from_pretrained(args.esm_model, low_cpu_mem_usage=True)

    model = model.cuda()

    # Uncomment to switch the stem to float16
    model.esm = model.esm.half()
    
    # Uncomment this line if your GPU memory is 16GB or less, or if you're folding longer (over 600 or so) sequences
    model.trunk.set_chunk_size(64)

    return model, tokenizer

def convert_outputs_to_pdb(outputs: dict):
    
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def tokenize_fasta(fasta: str, tokenizer: AutoTokenizer) -> torch.Tensor:

    tokenized_input = tokenizer([fasta], return_tensors="pt", add_special_tokens=False)['input_ids']
    
    # TODO: see later how we can use multiple gpus here, instead of just .cuda()
    inputs = tokenized_input.cuda()
    return inputs


def run_model(args, 
              model: EsmForProteinFolding, 
              tokenized: torch.Tensor) -> dict:
    
    with torch.no_grad():
        output = model(tokenized)
    
    return output

def main(args):
    """
    Main function of the script.
    """
    model, tokenizer = get_model(args)
    fastas_dict = parse_fasta_folder(args.fastas_folder)

    for i, (file, fastas) in enumerate(fastas_dict.items()):
        print("processing file currently: ", i, file)

        for j, (header, fasta) in enumerate(fastas.items()):
            print("current header: ", header)
            print("fasta: ", fasta)
            tokenized_input = tokenize_fasta(fasta, tokenizer)
            output_dict = run_model(args, model, tokenized_input)
            pdb = convert_outputs_to_pdb(output_dict)   
            with open(f"{args.output_folder}/{file}_{j}.pdb", "w") as f:
                f.write("".join(pdb))

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--esm_model", type=str, default="facebook/esmfold_v1", help="huggingface esmfold version")  
    argparser.add_argument("--fastas_folder", type=str, default="./fastas/", help="Folder to the fastas")
    argparser.add_argument("--output_folder", type=str, default="./fastas/", help="Folder to output the pdb files")    

    args = argparser.parse_args()    

    main(args)