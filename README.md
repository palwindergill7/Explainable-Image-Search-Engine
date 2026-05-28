# Explainable Image Search Engine

This project implements an explainable image search engine.

## Setup and Usage

Follow these steps to set up and run the application.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/palwindergill7/Explainable-Image-Search-Engine.git

```

### 2. Create and Activate the Conda Environment

Create the Conda environment using the provided `vision_gill.yml` file. This will install all the necessary dependencies.

```bash
conda env create -f vision_gill.yml
```

After the environment is created, activate it:

```bash
conda activate vision_gill
```

### 3. Prepare the Data

Place your images in the `Data/` folder.

### 4. Calculate Image Embeddings

Run the `Image_embeding_Calculator.ipynb` notebook to calculate the embeddings for your images. This will generate a `embeddings_cache.pt` file that the application uses for searching.

### 5. Run the Application

Finally, launch the Streamlit application:

```bash
cd Explainable-Image-Search-Engine
streamlit run app.py
```
