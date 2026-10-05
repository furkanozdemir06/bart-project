# San Francisco BART Ridership Analysis & Prediction

An end-to-end data analysis and machine learning project on BART (Bay Area Rapid Transit) ridership. It uses 13 million+ hourly station-to-station records from 2016 and 2017 to uncover traffic patterns and to predict passenger flow between stations.

## Project Overview

The project has two parts:

1. **Data Analysis:** answer practical questions about ridership (busiest station, least used route, peak days and hours, late-night usage, and the best time to find a seat from Berkeley to San Francisco).
2. **Data Science:** compute straight-line distances between stations and build a regression model that predicts the number of passengers traveling between two stations at a given hour and weekday.

## Dataset

| File | Content |
| :--- | :--- |
| `date-hour-soo-dest-2016.csv` | Hourly ridership between origin and destination stations, 2016 |
| `date-hour-soo-dest-2017.csv` | Same structure, 2017 |
| `station_info.csv` | Station names, addresses and coordinates |

After merging both years the dataset contains **13,285,207 rows** with no missing values.

## Methodology

1. **Data preparation:** merged the 2016 and 2017 files, converted timestamps to datetime, and extracted year, month, day, weekday name and hour.
2. **Exploratory analysis:** grouped and aggregated ridership by station, route, hour and weekday, and visualized the results with Matplotlib.
3. **Distance calculation:** parsed station coordinates and computed pairwise straight-line distances with the Haversine formula (scikit-learn), shown for the Berkeley and downtown San Francisco stations.
4. **Predictive modeling:** aggregated passengers by hour, origin, destination and weekday (301,922 rows), one-hot encoded the categorical features (92 features), and split the data 80/20 for training and testing.
5. **Model comparison:** trained six regression models and compared them using R².

## Key Findings

- **Busiest station:** Embarcadero (EMBR), counting both boardings and exits.
- **Least used route:** WSPR → SBRN, with only 40 passengers across two years.
- **Busiest day:** Wednesday (about 30.7M passengers), followed closely by Tuesday and Thursday. Weekend ridership is far lower (Sunday about 9.6M).
- **Rush hours:** Berkeley-to-SF ridership peaks at 8:00-9:00 AM and shows a second peak around 5:00-6:00 PM.
- **Late-night ridership (21:00-23:59):** about 11.7M passengers in total, dropping each hour (4.6M at 21:00, 4.0M at 22:00, 3.0M at 23:00).
- **Finding a seat from Berkeley to SF:** the morning commute is the most crowded period (more than 360,000 riders at both 8:00 and 9:00). Midday trips (13:00-16:00) and evenings after 19:00 carry far fewer riders, so they are the best windows for a seat. Traveling before 6:00 AM also works.

## Model Results

| Model | R² Score |
| :--- | :---: |
| **Extra Trees Regressor** | **0.965** |
| Decision Tree Regressor | 0.962 |
| XGBoost Regressor | 0.737 |
| Gradient Boosting Regressor | 0.449 |
| Ridge Regression | 0.172 |
| Lasso Regression | 0.170 |

Tree-based models clearly outperform linear models. Ridership depends on the interaction between route, hour and weekday, which linear models cannot capture well. The **Extra Trees Regressor** performed best with an R² of 0.965.

## Limitations and Next Steps

- The model predicts ridership from time and route only. Adding **weather, holidays, special events, employment data and population density** would likely improve accuracy and explain unusual days.
- The Haversine distance is not yet used as a model feature. Adding distance, travel time and station-level attributes could help the model generalize to new routes.
- Hyperparameter tuning and cross-validation would give a more reliable performance estimate than a single train/test split.
- Predictions like these could help transit agencies **schedule trains**, **plan maintenance in low-traffic windows**, and **manage crowding** at peak hours.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn

## Files

- `Bart_Project.ipynb`: full analysis and modeling notebook
