# Sussex Project

This project analyzes neural activity in mice using PCA, running windows, and recurrent neural networks (RNNs).

## Folder Structure

- `python code submitted/`: Jupyter notebooks and analysis scripts  
  - **Analysis Scripts**:  
    - `bel_pairs`: explores different types of potential clustering and their results  
    - `bel_RNN_*`: a series of notebooks showing results with different sampling methods:
      - **SMOTE**, **undersampling**, and **hybrid** strategies  
      - PCA/dimensionality reduction applied across **time** or **neural activity**
    - Most results are saved as CSV files (e.g., `test_accuracy_*.csv`)
    
- `.ipynb_checkpoints/`: Jupyter autosaves (ignored by Git)

## Notes

- Presentation with key results: `mice/Results_PPC_presentation.ppt`
