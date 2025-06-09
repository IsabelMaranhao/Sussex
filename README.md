# Sussex Project

This project analyzes neural activity in mice using PCA, running windows, and recurrent neural networks (RNNs).

## Folder Structure

- `python code submitted/`: Jupyter notebooks and analysis scripts  
  - **Analysis Scripts**:  There are many variations of the same analyis. The type used in described in each notebook.
    - `bel_pairs`: explores different types of potential clustering and their results  
    - `bel_RNN_*`: a series of notebooks showing results with different sampling methods:
      - **SMOTE**, **undersampling**, and **hybrid** strategies  
      - PCA/dimensionality reduction applied across **time** or **neural activity**
  - **input data**:
      go to 'python code submitted/processed_data' 
  - **Results**:
    - Most results are saved as CSV files (e.g., `test_accuracy_*.csv`)
    - Variations of running window analysis ar saved as 'running_results_[brain_area][type]' 
- `.ipynb_checkpoints/`: Jupyter autosaves 

## Notes

- Presentation with key results: `Results_PPC_presentation.ppt`
