if [ ! -d "myenv" ]; then
    python -m venv myenv
fi
source myenv/bin/activate
pip install -r requirements.txt
pip install plotly kaleido ipywidgets huggingface-hub pandas plotly
python src/DataLoader.py --prepare_all
python3 meddistillation_experiment.py
