import os
import pandas as pd

# Klasördeki tüm CSV dosyalarını listeleme
directory = 'dataset'
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Tüm CSV dosyalarını okuma ve birleştirme
combined_df = pd.DataFrame()

for csv_file in csv_files:
    csv_path = os.path.join(directory, csv_file)
    df = pd.read_csv(csv_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Birleştirilmiş DataFrame'i yeni bir CSV dosyasına kaydetme
combined_csv_path = os.path.join('tüm', 'combined_data.csv')
combined_df.to_csv(combined_csv_path, index=False)




