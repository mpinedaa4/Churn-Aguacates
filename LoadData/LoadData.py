import pandas as pd
from pathlib import Path
from typing import Optional

class LoadData:
    """Locates and loads the project dataset from the Dataset/ subdirectory."""
    def __init__(self) -> None:
        # Define the project root directory
        self.project_root: Path = Path(__file__).resolve().parent.parent
        # Define the path to the CSV dataset
        self.file_path: Path = self.project_root / "Dataset" / "dataset.csv"
        # Initialize the Dataframe to None
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """ Load the dataset from a CSV file and return it as a DataFrame. """

        # If the file does not exist
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {self.file_path}\n"
                "Ensure 'dataset.csv' is placed inside the 'Dataset/' folder."
            )

        print(f"Loading file: {self.file_path}")

        self.df = pd.read_csv(self.file_path) # Reads the datasset and transforms it into a dataframe

        print(
            f"Dataset loaded successfully."
            f"Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns."
        )

        return self.df