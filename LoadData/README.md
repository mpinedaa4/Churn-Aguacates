# LoadData

## Purpose

Locates and loads the raw CSV dataset into a pandas DataFrame. All other modules receive their data from this class rather than reading the file themselves, ensuring a single point of file I/O across the entire pipeline.

---

## File

`LoadData/LoadData.py`

---

## Class: `LoadData`

### Constructor

```python
loader = LoadData()
```

Resolves the project root directory automatically using `Path(__file__).resolve().parent.parent`, then constructs the expected path to `Dataset/dataset.csv`. No arguments are required.

### Method: `load_data() -> pd.DataFrame`

Reads the CSV file and returns it as a pandas DataFrame. Raises `FileNotFoundError` with a descriptive message if the file is not found at the expected path.

```python
df = loader.load_data()
```

**Returns:** A raw `pd.DataFrame` with no transformations applied.

**Raises:** `FileNotFoundError` if `Dataset/dataset.csv` does not exist.

---

## Expected File Location

The dataset must be placed at:

```
project_root/
`-- Dataset/
    `-- dataset.csv
```

The path is resolved relative to the location of `LoadData.py`, so it is independent of the working directory from which the script is executed.

---

## Usage in the Pipeline

`LoadData` is instantiated inside `EDA.__init__()`. It is not called directly from `main.py`.

```python
# Inside EDA/EDA.py
from LoadData.LoadData import LoadData

loader = LoadData()
self.df = loader.load_data()
```