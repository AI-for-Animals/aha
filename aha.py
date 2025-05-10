#!/usr/bin/env python3
"""
Main module for the AHA evaluation.
"""
import argparse, logging, os, random, time, json
from pathlib import Path
from typing import List, Dict, Any, Set
import pandas as pd

import nest_asyncio
nest_asyncio.apply()

from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import Model, get_model, GenerateConfig
from inspect_ai.solver import generate
from scoring import SOLVER_INSTRUCTIONS, final_digit_model_graded_qa
from utils import (
    setup_logging, mkd, read_json, write_json,
    get_latest_file, combine_csv_files
)

setup_logging(logging.INFO)
logging.getLogger("inspect_ai.model").setLevel(logging.INFO)
config = None
dataset_path = None

class Config:
    def __init__(self, args):
        self.model = args.model
        self.judges = args.judges.split(',')
        self.openai_base_url = args.openai_base_url
        self.dataset_path = Path(args.dataset) if args.dataset else Path("data_public.json")
        self.output_dir = Path(args.output_dir) if args.output_dir else Path("results")
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.shuffle = args.shuffle
        self.model_temperature = None if args.model_temperature.lower()=="none" else float(args.model_temperature)
        self.judge_temperature = None if args.judge_temperature.lower()=="none" else float(args.judge_temperature)
        self.run_analysis = args.run_analysis
        self.num_batches = args.num_batches
        self.start_batch = args.start_batch
        self.current_batch = 1
        self.calculate_judge_harshness = args.calculate_judge_harshness
        self.model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}

def setup_environment(conf: Config) -> None:
    mkd(conf.output_dir)
    random.seed(conf.seed)

    os.environ["INSPECT_EVAL_MODEL"] = conf.model
    logging.info(f"INSPECT_EVAL_MODEL set to: '{conf.model}'")
    logging.info(f"AHA Config: Judges='{','.join(conf.judges)}'")
    logging.info(f"AHA Config: Batches={conf.num_batches}x{conf.batch_size}, Seed={conf.seed}")
    logging.info(f"AHA Config: Temp(Model)={conf.model_temperature}, Temp(Judge)={conf.judge_temperature}")

    if conf.model.startswith("vllm/"):
        logging.info("vLLM provider detected. Setting VLLM environment variables for generation params...")

        # Keep temperature and seed env vars for now, though they might be redundant
        # if generate() passes them correctly.
        if conf.model_temperature is not None:
             os.environ["VLLM_TEMPERATURE"] = str(conf.model_temperature)
        if conf.seed is not None:
             os.environ["VLLM_SEED"] = str(conf.seed)

        # Keep TP_SIZE and MAX_TOKENS as they might affect initialization or defaults
        os.environ.setdefault("VLLM_TP_SIZE", "1")
        os.environ.setdefault("VLLM_MAX_TOKENS", "1000")

        vllm_vars = {k: os.environ[k] for k in os.environ if k.startswith("VLLM_")}
        logging.info(f"Remaining VLLM environment variables: {vllm_vars}")

    elif conf.model.startswith("hf/"):
        logging.info("Hugging Face (hf) provider detected. Setting HF environment variables...")
        # os.environ.setdefault("HF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        hf_vars = {k: os.environ[k] for k in os.environ if k.startswith("HF_")}
        logging.info(f"HF environment variables: {hf_vars}")

    if conf.openai_base_url:
         logging.info(f"Setting OPENAI_BASE_URL for potential use by providers (e.g., judges): {conf.openai_base_url}")
         os.environ["OPENAI_BASE_URL"] = conf.openai_base_url
         os.environ.setdefault("OPENAI_API_KEY", "DUMMY_KEY_FOR_ENDPOINT")

    logging.info(f"AHA: {conf.num_batches}x{conf.batch_size}, model={conf.model}")
    logging.info(f"temp={conf.model_temperature}, judge_temp={conf.judge_temperature}, seed={conf.seed}")

def load_and_sample_data(conf: Config, data: List[Dict[str,Any]], used: Set[int]) -> Path:
    if conf.shuffle:
        avail = set(range(len(data))) - used
        if len(avail) < conf.batch_size:
            used.clear(); avail = set(range(len(data)))
        chosen = random.sample(list(avail), conf.batch_size)
    else:
        start = (conf.start_batch + conf.current_batch - 1) * conf.batch_size
        if start >= len(data):
            start = 0; used.clear()
        chosen = list(range(start, min(start + conf.batch_size, len(data))))
    
    used.update(chosen)
    sampled = [data[i] for i in chosen]
    
    tdir = conf.output_dir/"temp"; mkd(tdir)
    sfile = tdir/f"sampled_data_batch{conf.current_batch}.json"
    write_json(sfile, sampled)
    return sfile

def record_to_sample(record: Dict[str,Any]):
    meta = record.get("generation_tags", {}).copy()
    meta["sample_id"] = record["sample_id"]
    meta["tags"] = record.get("tags", [])
    return Sample(
        input=f"{record['question']} Your answer must be EXACTLY 30–40 words long.",
        target=SOLVER_INSTRUCTIONS,
        metadata=meta
    )

@task
def aha_evaluation() -> Task:
    try:
        from __main__ import config, dataset_path
    except ImportError:
        global config, dataset_path
    
    ds = json_dataset(str(dataset_path), sample_fields=record_to_sample)
    temp = generate(max_tokens=1000, temperature=config.model_temperature, seed=config.seed, cache=True) if (config.model_temperature and config.seed) else generate(max_tokens=1000, cache=True)
    return Task(
        dataset=ds,
        solver=temp,
        scorer=final_digit_model_graded_qa(config.judges, judge_temperature=config.judge_temperature),
        max_retries=3,
        max_messages=20,
        num_batches=1
    )

def combine_csv_results(conf: Config) -> None:
    cfiles = sorted(conf.output_dir.glob("results_*.csv"))
    if not cfiles:
        logging.error("No CSV files found to combine.")
        return
    df = combine_csv_files(cfiles)
    cols = list(df.columns)
    answer_cols = [c for c in cols if c.endswith("_answer")]
    tag_cols = [c for c in cols if c.startswith("tag")]
    other_cols = [c for c in cols if c not in answer_cols+tag_cols]
    front_cols = []
    for c in ["sample_id","input"]:
        if c in other_cols:
            front_cols.append(c)
            other_cols.remove(c)
    order = front_cols + answer_cols + other_cols + tag_cols
    df = df[order]
    outp = conf.output_dir/"combined_results.csv"
    df.to_csv(outp, index=False)
    logging.info(f"Combined CSV saved to: {outp}")

def main():
    p = argparse.ArgumentParser("AHA Benchmark")
    p.add_argument("--model", required=True)
    p.add_argument("--openai-base-url", 
                  help="Base URL for OpenAI-compatible endpoints (e.g., HuggingFace inference endpoints)")
    p.add_argument("--judges", default="anthropic/claude-3-5-sonnet-20241022,google/gemini-1.5-pro-002,openai/gpt-4o-2024-08-06")
    p.add_argument("--dataset")
    p.add_argument("--output_dir")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--num_batches", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--model_temperature", default="0.5")
    p.add_argument("--judge_temperature", default="0.5")
    p.add_argument("--run-analysis", action="store_true")
    p.add_argument("--start-batch", type=int, default=0)
    p.add_argument("--calculate_judge_harshness", action="store_true", help="If set, judge harshness is calculated from data.")
    p.add_argument("--model-kwargs", type=str, default=None,
                  help="JSON string of additional kwargs for model initialization")
    args = p.parse_args()

    global config, dataset_path
    config = Config(args)
    os.environ["INSPECT_EVAL_MODEL"] = config.model # Set model name globally
    setup_environment(config) # Setup other env vars

    try:
        full_data = read_json(config.dataset_path)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    used_indices = set()
    sampled_files = []
    start_time = time.time()

    try:
        for b in range(config.num_batches):
            config.current_batch = b + 1
            
            sfile = load_and_sample_data(config, full_data, used_indices)
            sampled_files.append(sfile)
            dataset_path = sfile
            
            # --- Create model_args dictionary ---
            model_args_dict = {}
            if config.model.startswith("vllm/"):
                logging.info("Preparing model_args for vLLM provider...")
                model_args_dict = {
                    "max_model_len": 8192,
                    "gpu_memory_utilization": 0.95,
                    "trust_remote_code": True
                }
                logging.info(f"Passing model_args to eval(): {model_args_dict}")
            # ------------------------------------

            logging.info(f"Processing batch {config.current_batch}/{config.num_batches} using dataset: {dataset_path}")
            try:
                 # Pass model_args dictionary to eval()
                 eval_result = eval("aha_evaluation", model_args=model_args_dict)
                 logging.info(f"Finished evaluation for batch {config.current_batch}.")
            except Exception as e:
                logging.error(f"Error evaluating batch {config.current_batch}: {e}")
                continue
            
            if config.run_analysis:
                ldir = Path("./logs")
                latest = get_latest_file(ldir, "*.eval")
                if latest:
                    cmd = (
                        f"python analysis.py "
                        f"--log-file {latest} "
                        f"--output-dir {config.output_dir} "
                        f"--solver-name '{config.model}'"
                    )
                    if config.calculate_judge_harshness:
                        cmd += " --calculate_judge_harshness"
                    logging.info(f"Results for batch {config.current_batch}:")
                    os.system(cmd)
        
        dt = time.time() - start_time
        logging.info(f"Total time: {dt:.2f}s")
        
        if config.run_analysis:
            logging.info("Combining results...")
            combine_csv_results(config)
            cpath = config.output_dir/"combined_results.csv"
            if cpath.exists():
                logging.info("Final analysis on combined_results.csv...")
                cmd = (
                    f"python analysis.py "
                    f"--csv-file {cpath} "
                    f"--output-dir {config.output_dir}"
                )
                if config.calculate_judge_harshness:
                    cmd += " --calculate_judge_harshness"
                os.system(cmd)
            else:
                logging.warning("No combined CSV found.")
    
    finally:
        for f in sampled_files:
            if f.exists():
                f.unlink()

if __name__ == "__main__":
    main()
