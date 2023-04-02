from tkinter import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np  # linear algebra
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from PIL import Image, ImageTk
import joblib


train = pd.read_csv("train_data.csv")
test = pd.read_csv("test.csv")
final = pd.read_csv("test.csv")
df = pd.concat([train, test])
df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")


def cols_new(data_df):
    data_df['year'] = data_df['date'].dt.year
    data_df['quarter'] = data_df['date'].dt.quarter
    data_df['month'] = data_df['date'].dt.month
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    data_df['dayofweek'] = data_df['date'].dt.dayofweek
    return data_df


cols_new(df)
df.groupby(['item', 'store'])['sales'].median()


def mean_cols(data, cols):
    for i in cols:
        cols = [e for e in cols if e not in (i)]
        for j in cols:
            if i != j:
                data['mean_'+i+'_' +
                     j] = data.groupby([i, j])['sales'].transform('mean')
    return data


mean_cols(df, ['item', 'store', 'dayofweek', 'weekofyear', 'month', 'quarter'])
# print(df.columns)


def median_cols(data, cols):
    for i in cols:
        cols = [e for e in cols if e not in (i)]
        for j in cols:
            if i != j:
                data['median_'+i+'_' +
                     j] = data.groupby([i, j])['sales'].transform('median')
    return data


median_cols(df, ['item', 'store', 'dayofweek',
            'weekofyear', 'month', 'quarter'])
# print(df.columns)
train = df.loc[~df.sales.isna()]
test = df.loc[df.sales.isna()]

X_train = train.drop(['date', 'sales', 'id'], axis=1)
y_train = train['sales'].values
X_test = test.drop(['id', 'date', 'sales'], axis=1)
x_train, x_validate, y_train, y_validate = train_test_split(
    X_train, y_train, random_state=100, test_size=0.25)

model = joblib.load(
    "C:\\Users\\kumar\\OneDrive\\Desktop\\demand_forecast\\xgbmodel_model.joblib")
predict = model.predict(xgb.DMatrix(
    X_test), ntree_limit=model.best_ntree_limit)
final['sales'] = np.round(predict)


splash_root = Tk()
splash_root.title("Demand Forecast")
splash_root.geometry("850x500")
image = Image.open("demand4.png")
photo = ImageTk.PhotoImage(image)

# Create a Label widget to display the image
label = Label(splash_root, image=photo)
label.pack()
image = None


def main():
    # destroy splash window
    splash_root.destroy()

    # Execute tkinter
    root = Tk()
    root.title("Demand Forecast")
    # Adjust size
    root.geometry("850x500")

    canvas1 = Canvas(root, width=850, height=200, bg='lavender')
    canvas1.pack()

    label1 = Label(root, text='Date: ', bg='lavender')
    canvas1.create_window(350, 30, window=label1)

    entry1 = Entry(root)  # create 1st entry box
    canvas1.create_window(450, 30, window=entry1)

    # New_Unemployment_Rate label and input box
    label2 = Label(root, text=' Store: ', bg='lavender')
    canvas1.create_window(350, 60, window=label2)

    entry2 = Entry(root)  # create 2nd entry box
    canvas1.create_window(450, 60, window=entry2)

    label3 = Label(root, text=' Item: ', bg='lavender')
    canvas1.create_window(350, 90, window=label3)

    entry3 = Entry(root)  # create 3rd entry box
    canvas1.create_window(450, 90, window=entry3)
    canvas1.pack()

    def predict_sales(date, store, item):
        # sample= np.array([id,sales]).reshape(1,-1)
        # print(predict)
        ans = final.groupby(['date', 'item', 'store'])['sales'].mean()
        return (ans.loc[(date, item, store)])

    def values():
        global date1  # our 1st input variable
        date1 = str(entry1.get())

        global store1  # our 2nd input variable
        store1 = float(entry2.get())

        global item1
        item1 = float(entry3.get())

        Prediction_result = ("Predicted Sales:",
                             predict_sales(date1, store1, item1))
        label_Prediction = Label(root, text=Prediction_result, bg='salmon')
        canvas1.create_window(445, 170, window=label_Prediction)

    # button to call the 'values' command above
    button1 = Button(root, text='Predict Sales',
                     command=values, bg='lightgreen')
    canvas1.create_window(445, 130, window=button1)

    figure1 = plt.Figure(figsize=(5, 4), dpi=90)  # width,height
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().pack(side=LEFT)
    df1 = train[['sales', 'store']].groupby('store').max()
    df1.plot(kind='bar', legend=True, ax=ax1)
    ax1.set_title('Max Sales per store')

    figure2 = plt.Figure(figsize=(5, 4), dpi=90)
    ax2 = figure2.add_subplot(111)
    bar2 = FigureCanvasTkAgg(figure2, root)
    bar2.get_tk_widget().pack(side=RIGHT)
    df2 = train[['sales', 'store']].groupby('store').mean()
    df2.plot(kind='bar', legend=True, ax=ax2, color='r')
    ax2.set_title('Avg Sales per store')


splash_root.after(5000, main)


mainloop()
