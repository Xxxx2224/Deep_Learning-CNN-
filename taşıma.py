import os
import shutil

# Kaynak ve hedef klasörleri tanımla
source_directory = 'dataset'


# Klasördeki tüm dosyaları listele
files = os.listdir(source_directory)

# `.csv` uzantılı dosyaları hedef klasöre taşı
for file in files:
    if file.endswith('.csv'):
        source_file = os.path.join(source_directory, file)
        destination_file = os.path.join('tümscv', file)
        shutil.move(source_file, 'tümcsv')
        print(f"{file} taşındı.")
