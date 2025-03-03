
# Downloading virtual environment

python -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate  # Windows


# Git install needed libraries

pip install streamlit librosa numpy matplotlib pandas soundfile


# Create empty Python file

touch app.py


# Run the App with this command in Git 

streamlit run app.py

