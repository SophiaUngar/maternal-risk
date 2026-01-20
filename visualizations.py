import Load_Data
import matplotlib.pyplot as plt

def visualize_variables():
    filename = "Data/Maternal Health Risk Data Set.csv"  # data filename
    y_name = 'RiskLevel'  # name of category we are trying to predict

    x, y = load_data.process(filename, y_name)



visualize_variables()