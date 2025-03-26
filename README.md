# Median Housing Value Prediction

The housing data can be downloaded from:  
[Housing Data](https://raw.githubusercontent.com/ageron/handson-ml/master/)  

The script includes code to download the data. We have modeled the median house value using the given housing data.

## Techniques Used
- **Linear Regression**
- **Decision Tree**
- **Random Forest**

## Steps Performed
1. **Data Preparation & Cleaning**  
   - Checked and imputed missing values.  

2. **Feature Engineering & Correlation Analysis**  
   - Generated features and examined variable correlations.  

3. **Sampling & Splitting**  
   - Evaluated multiple sampling techniques.  
   - Split the dataset into training and test sets.  

4. **Modeling & Evaluation**  
   - Trained and evaluated models using the above techniques.  
   - **Final evaluation metric:** Mean Squared Error (MSE).  

## Environment Setup
### Command to Create an Environment from `environment.yml`
```bash
conda create -f env.yaml
```
### Activate the Conda Environment
```bash
conda activate mle-dev
```
### To excute the python script
```bash
python nonstandardcode.py
```
