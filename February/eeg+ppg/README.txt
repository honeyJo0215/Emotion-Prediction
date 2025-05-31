Multi-modal emotion classification model based on DEAP dataset (EEG, PPG)
- Each subject (s01 to s22) uses a data file that has already been preprocessed and saved.
- EEG data: (40, 32, 128) → Model input: (40, 32, 128, 1)
- PPG data: (40, 128) → Model input: (40, 128, 1, 1)
- 라벨: (40,) (Negative: 0, Positive: 1, Neutral: 2)
