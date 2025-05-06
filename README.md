# Animal Harm Assessment (AHA) Benchmark

The **Animal Harm Assessment (AHA) Benchmark** is an evaluation framework designed to assess the potential for harm to nonhuman animals in generated text by large language models (LLMs). It combines questions retrieved from reddit and synthetic questions (based on animal ethics literature).

For more information, see preprint: 
https://arxiv.org/abs/2503.04804 

---

## Overview

AHA benchmark investigates how LLM-generated text can impact animal welfare by evaluating responses to a diverse set of questions. The original set of questions include:
- **1,850 curated Reddit questions** (filtered for relevance to animal ethics), and
- **2,500 synthetic questions** created by combining 50 animal categories with 50 ethical scenarios.

A stratified sample of 3045 questions (70% of 4350) is used as a public test set (data_public.json). The remaining 1305 (30%) questions are retained privately to track and test for contamination.

The evaluation framework uses LLMs-as-judges with a three-level scoring system:
- `[-1]` – Increases risk of harm
- `[0]`  – No clear effect or uncertain
- `[1]`  – Decreases risk of harm

Full details of the scoring rubric are in scoring.py.

---

## Setup Instructions

### Example Usage in Google Colab

Follow these steps in a Google Colab notebook to run the AHA benchmark:

1. **Clone the Repository**

   ```
   # 1. Clone the files
   !git clone https://github.com/AI-for-Animals/aha.git aha
   %cd /content/aha 
   
   # 2.1 install dependencies. If necessary, restart session.
   !pip install inspect-ai anthropic google-generativeai openai
   !pip install --upgrade google-genai

   # 2.2 Retrieve API keys. The three API keys needed to use the default LLMs-as-judges. Specify fewer/other judges if needed (see how in 3.2).
   import os
   from google.colab import userdata
   os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
   os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
   os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

   # 3. Run examples
   # 3.1 A simple example. The'--run-analysis' option saves & combines results in csv files.
   # The default three LLMs-as-judges are: anthropic/claude-3-5-sonnet-20241022,google/gemini-1.5-pro-002,openai/gpt-4o-2024-08-06
   !python aha.py --model 'anthropic/claude-3-5-haiku-20241022' --batch_size 2 --run-analysis
   
   # 3.2 A longer example with more options explicitly set (see aha.py for all options):
   !python aha.py \
   --model 'openai/gpt-4o-mini-2024-07-18' \
   --judges 'anthropic/claude-3-5-haiku-20241022,google/gemini-1.5-flash-002,openai/gpt-4o-mini-2024-07-18' \
   --batch_size 2 --num_batches 2 \
   --seed 0 --model_temperature 1 --judge_temperature 0 \
   --run-analysis
   
   # 4. Standard evaluations (uncomment to run) 
   
   # 4.1 Small sample (default, 100 questions)
   # !python aha.py --model 'anthropic/claude-3-5-haiku-20241022' --shuffle --run-analysis
   
   # 4.2 Full sample (3045 questions)
   # !python aha.py --model 'anthropic/claude-3-5-haiku-20241022' --batch_size 435 --num_batches 7 --run-analysis

   # For additional analysis, including self-preference adjustments for LLMs-as-judges, and related models:
   # !pip install krippendorff
   # !python results_summary_raw.py
   # !python results_summary.py

   # More scripts for analysis:
   # !python tag_analysis.py
   # !python figures_final.py #NB! Uses hardcoded values.
   # !python pairwise_comparison.py #NB!BETA. Observed to work correctly with identically formated separate input files per model.  

   # It can be useful to store and retrieve files remotely:
   # from google.colab import drive
   # drive.mount('/content/drive')
   ```

## Project Structure
aha.py – Main evaluation script.
scoring.py – Module implementing the LLM-as-a-judge scoring function.
analysis.py – Script for combining CSV results and analyzing benchmark outputs.
utils.py – Shared utility functions for logging, file I/O, CSV operations, and timestamp handling.
data_public.json - input data.

## Requirements
Default LLMs-as-judges are Anthropic, Google, and OpenAI models. 
Dependencies: inspect-ai anthropic google-generativeai openai.
API Keys: Required for Anthropic, Google, and OpenAI.

## License
This project is licensed under the MIT License.

## Acknowledgments
Development of this benchmark has been supported by Hive / AI for Animals. The setup has been built on UK AI Safety Institute's Inspect Evals.
