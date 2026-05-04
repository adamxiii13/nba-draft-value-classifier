from sklearn.model_selection import GridSearchCV

# Define the "menu" of settings for the AI to try
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}

# This will run the model over and over until it finds the best combo
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1_macro')
grid_search.fit(X_train_res, y_train_res)

print(f"Best Settings Found: {grid_search.best_params_}")