python -m venv .
source./bin/activate
pip install -r requirements.txt
pip install plotly kaleido ipywidgets huggingface-hub pandas plotly
python src/DataLoader.py --prepare_all
python3 meddistillation_experiment.py
