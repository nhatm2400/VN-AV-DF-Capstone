"""STEP 01: Fill videos.csv with selected URLs, then Run to build selected_videos.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation._run import run
from src.data.preparation.settings import SOURCES

if __name__ == '__main__':
    run('src.data.preparation.collect', ['--input', SOURCES / 'videos.csv', '--out', SOURCES / 'selected_videos.csv'])
