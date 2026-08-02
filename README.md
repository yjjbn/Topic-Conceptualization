# Topic-Conceptualization

In "toy":  

to recreate environment:  
uv sync --locked  
have a file in top directory called .env with API key OPENROUTER_API_KEY=[CODE]  
  
01_codebook_dev  
./data contains data used to develop codebook  
./prompts prompts for classifying docs and requesting codebook  
./out codebooks and other results output here  
  
to classify docs and request codebook, run in 01_codebook_dev:  
uv run python initial_classify.py --model [MODEL NAME]  
-> will output "initial_classification" JSONs -> this is passed to model to write "codebook" JSONs  
  
02_experiment  
./data contains data used to test codebook  
./prompts prompts for classifying docs  
./out classification results output here  
  
to apply the codebook, run in 02_experiment:  
uv run python apply_codebook.py --model [MODEL NAME] --codebook_path "../01_codebook_dev/out/[NAME OF CODEBOOK].json"  
-> will output csv with classifications  
