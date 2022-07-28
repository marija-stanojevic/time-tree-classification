import glob
import pandas as pd
import textract
import ocrmypdf as ocr1
import re
import os
from utils import clean_ids, normalize_text_in_column
import tarfile
import zipfile
import shutil


def unzip():
    dirs_with_zipped_data = ['iGEM', 'google_scholar', 'journals', 'PMC', 'PMC_historical', 'biorxivTDM']
    for directory in dirs_with_zipped_data:
        for file in os.listdir(directory):
            if file.endswith('.zip'):
                path = os.path.join(directory, file)
                with zipfile.ZipFile(path) as zip_ref:
                    zip_ref.extractall()


def unzip_pmc_and_historical_after_downloading():
    dirs_with_tar_data = ['PMC', 'PMC_historical']
    for directory in dirs_with_tar_data:
        extracted = []
        files = os.listdir(directory)
        files.reverse()
        for file in files:
            if file.endswith('.tar.gz'):
                with tarfile.open(os.path.join(directory, file), "r:gz") as tar_ref:
                    for member in tar_ref.getmembers():
                        member.name = member.name.split('/')[-1]
                        extracted.append(member.name)
                        tar_ref.extract(member, os.path.join(directory, 'text_papers'))


def process_igem_data():
    csv_name = 'iGEM/iGEMsPDFs_metadata.csv'
    folder = 'iGEM/iGEMsPDFs/'
    df = pd.read_csv(csv_name)
    df = df.drop_duplicates(keep=False)
    df = df.reset_index(drop=True)
    df['downloaded'] = [0] * df.shape[0]
    df['text'] = [''] * df.shape[0]
    for file in glob.glob(folder + '*.pdf'):
        id = file.split('/')[2].replace('.pdf', '')
        try:
            if df[df['ID'] == id].shape[0] > 0:
                row = df[df['ID'] == id]
                # Since some iGEM file names are numbers, there is an issue in future processing
                # We put quotes around filename to show that it should always be seen as a string
                row['ID'] = "\"" + row['ID'] + "\""
                try:
                    ocr1.ocr(file, folder + "output.pdf", deskew=True, force_ocr=True)
                    text = textract.process(folder + "output.pdf", method='pdftotext').decode('utf-8')
                except Exception:
                    text = textract.process(file, method='pdftotext').decode('utf-8')
                row['text'] = text
                if len(text) > 100:
                    row['downloaded'] = 1
                row = row[['ID', 'text', 'downloaded', 'Binary']]
                row.to_csv('fulltext_corpus/text' + csv_name.split('/')[1].replace('_metadata', ''), mode='a',
                           quotechar='"', index=False, header=False)
            else:
                os.remove(file)
                print('File deleted: ' + file)

        except Exception as e:
            print(id, df[df['ID'] == id]['Binary'].tolist()[0], e)


def process_crawled_data(csv_name, folder, pdf):
    print(folder)
    df = pd.read_csv(csv_name)
    df = clean_ids(df)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df['downloaded'] = [0] * df.shape[0]
    df['text'] = [''] * df.shape[0]
    for file in glob.glob(folder + '*.pdf'):
        id = re.sub(r'[^\s\w]+', ' ', file[file.index('/') + 1: file.index('.pdf')]).replace('  ', ' ').lower() + pdf
        try:
            if df[df['ID'] == id].shape[0] > 0:
                row = df[df['ID'] == id]
                if df.at[df.index[df['ID'] == id][0], 'downloaded'] == 1:  # remove duplicate files
                    print('Duplicate deleted: ' + file)
                    os.remove(file)
                    continue
                try:
                    ocr1.ocr(file, folder + "output.pdf", deskew=True, force_ocr=True)
                    text = textract.process(folder + "output.pdf", method='pdftotext').encode('utf-8')
                except Exception:
                    text = textract.process(file, method='pdftotext').decode('utf-8')
                row['text'] = text
                if len(text) > 100:
                    row['downloaded'] = 1
                    df.at[df.index[df['ID'] == id], 'downloaded'] = 1
                row = row[['ID', 'text', 'downloaded']]
                row.to_csv('fulltext_corpus/text' + csv_name.split('/')[1].replace('_metadata', ''), mode='a',
                           quotechar='"', index=False, header=False)
            else:  # remove files which are not in meta data
                os.remove(file)
                print('File deleted: ' + file)
        except Exception as e:
            print(id, e)


def is_phylogenetics_text(text):
    text = text.lower().replace('-', ' ')
    phylogenetics_words = [' mrbayes', ' beast', ' timetree', ' treeannotator', ' phylogen', ' beauti', ' reltime',
                           ' multidivtime', 'r8s', ' treefinder', 'path o gen', ' phytime', ' time tree', ' phylogram',
                           ' relaxed clock', ' mega', ' molecular clock', ' strict clock']
    for word in phylogenetics_words:
        if word in text:
            return True
    return False


def process_biorxiv_tdm(csv_name, folder, new_folder, filename):
    df = pd.read_csv(os.path.join(folder, csv_name))
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df['downloaded'] = [0] * df.shape[0]
    df['text'] = [''] * df.shape[0]
    for file in glob.glob(os.path.join(folder, '*.pdf')):
        id = file.split('/')[-1].replace('.full.pdf', '')
        version = int(id.split('v')[1])
        id = '10.1101/' + id.split('v')[0]
        try:
            if df[(df['ID'] == id) & (df['Version'] == version)].shape[0] == 1:
                row = df[(df['ID'] == id) & (df['Version'] == version)].copy()
                text = textract.process(file, method='pdftotext').decode('utf-8')
                if is_phylogenetics_text(text):
                    row['text'] = text
                    if len(text) > 100:
                        row['downloaded'] = 1
                        row = row[['ID', 'text', 'downloaded']]
                        row.to_csv(filename, mode='a', quotechar='"', index=False, header=False)
                else:  # remove files which are not related to phylogenetics
                    os.remove(file)
                    print('File deleted: ' + file)
        except Exception as e:
            print(id, e)
    shutil.move(folder, new_folder)


def process_pmc(folder, new_folder, csv_file):
    filename = os.path.join(new_folder, csv_file)
    for file in glob.glob(os.path.join(folder, '*.txt')):
        id = file.split('/')[-1].replace('.txt', '')
        try:
            with open(file, 'r', encoding='latin1') as text_file:
                text = text_file.read()
                if is_phylogenetics_text(text):
                    row = pd.DataFrame([[id, text, 1]], columns=['ID', 'text', 'downloaded'])
                    row.to_csv(filename, mode='a', quotechar='"', index=False, header=False)
                else:  # remove files which are not related to phylogenetics
                    os.remove(file)
                    print('File deleted: ' + file)
        except Exception as e:
            print(id, e)
    shutil.move(folder, new_folder)


def remove_non_phylogenetics_articles():
    for file in glob.glob('fulltext_corpus/text*.csv'):
        df = pd.read_csv(file, header=None)
        print(file, df.shape[0])
        indices_drop = []
        for index, row in df.iterrows():
            if not is_phylogenetics_text(row[1]):
                indices_drop.append(index)
        df = df.drop(labels=indices_drop, axis=0)
        print(file, len(indices_drop))
        df.to_csv(file, quotechar='"', header=False, index=False)


def get_mask(df):
    mask = []
    for index, row in df.iterrows():
        if type(row[1]) != float and len(row[1]) > 200:
            mask.append(index)
    return mask


def remove_duplicates():
    for file in glob.glob('fulltext_corpus/text*.csv'):
        print(file)
        df1 = pd.read_csv(file, header=None)
        print(len(df1))
        print(len(df1[df1[2] == 1]))
        df1 = df1.loc[get_mask(df1)]
        df1 = df1.drop_duplicates(keep='first')
        print(len(df1))
        df1.to_csv(file, quotechar='"', header=False, index=False)


def training_test_full_texts(labeled_file, crawler_file, igem_file):
    df = clean_ids(pd.read_csv(labeled_file))
    data = clean_ids(pd.read_csv(igem_file, header=None)[[0, 1, 2]], False)
    data2 = pd.read_csv(crawler_file, header=None)
    data = pd.concat([data, data2])
    data = data.merge(df, left_on=0, right_on='ID')[['ID', 'PY', 1, 'Binary']]
    data = normalize_text_in_column(data, 1)
    data = data.sample(frac=1)
    df_test = data[0:int(0.2 * data.shape[0])]
    df_training = data[int(0.2 * data.shape[0]):]
    df_test.to_csv('test.csv', quotechar='"', index=False, header=False)
    df_training.to_csv('train.csv', quotechar='"', index=False, header=False)


os.chdir('data')
# Unzips files from different datasets
# unzip()
# unzip_pmc_and_historical_after_downloading()

# Paths might need to be changed based on where you downloaded new data; also make sure the code works well before you
# run it as it deletes files and you don't want it to delete files you just downloaded because something is not working
# as intended
# process_igem_data()
# process_crawled_data('google_scholar/crawledData_metadata.csv', 'google_scholar/downloadedPDFs/', '')
# process_crawled_data('google_scholar/crawledDataRound2_metadata.csv', 'google_scholar/downloadedPDFsRound2/', '')
# process_crawled_data('journals/GeneJournalData_metadata.csv', 'journals/downloadedScienceDirectGenePDFs/', ' pdf')
# process_crawled_data('journals/MolecularJournalData_metadata.csv', 'journals/scienceDirectMolecularPdfs/', ' pdf')
# process_crawled_data('journals/WileyBiogeographyJournalData_metadata.csv', 'journals/downloadedWileyBiogeography/',
#                      '')
# process_crawled_data('journals/WileyCladisticsJournalData_metadata.csv', 'journals/downloadedWileyCladistics/', '')
# process_crawled_data('journals/WileyEcologyEvolutionJournalData_metadata.csv',
#                      'journals/downloadedWileyEcologyEvolution/', '')
# process_crawled_data('journals/WileyMolecularEcologyJournalData_metadata.csv',
#                      'journals/downloadedWileyMolecularEcology/', '')
# process_crawled_data('journals/WileyEvolutionaryBiologyJournalData_metadata.csv',
#                      'journals/downloadedWileyEvolutionaryBiology/', '')
# process_crawled_data('journals/WileyOrganicEvolutionJournalData_metadata.csv',
#                      'journals/downloadedWileyOrganicEvolution/', '')
# process_crawled_data('journals/WileyPhytologistTrustJournalData_metadata.csv',
#                      'journals/downloadedWileyPhytologistTrust/', '')
# process_crawled_data('journals/WileySystematicEntomologyJournalData_metadata.csv',
#                      'journals/downloadedWileySystematicEntomology/', '')
# process_crawled_data('journals/WileyZoologicalResearchJournalData_metadata.csv',
#                      'journals/downloadedWileyZoologicalResearch/', '')
# process_crawled_data('journals/OABiologicalLinneanJournalData_metadata.csv',
#                      'journals/downloadedOABiologicalLinnean/', '')
# process_crawled_data('journals/OABotanicalLinneanJournalData_metadata.csv', 'journals/downloadedOABotanicalLinnean/',
#                      '')
# process_crawled_data('journals/OABotanyAnnalsJournalData_metadata.csv', 'journals/downloadedOABotanyAnnals/', '')
# process_crawled_data('journals/OASystematicBiologyJournalData_metadata.csv',
#                      'journals/downloadedOASystematicBiology/', '')
# process_crawled_data('journals/OAZoologicalLinneanJournalData_metadata.csv',
#                      'journals/downloadedOAZoologicalLinnean/', '')
# process_biorxiv_tdm('biorxivTDM/biorxivTDM_metadata.csv', 'biorxivTDM', 'biorxivTDM',
#                    'fulltext_corpus/textBiorxivTDM.csv')
# process_pmc('text_papers', 'PMC', 'fulltext_corpus/textPMC.csv')
# process_pmc('PMC_historical_text_papers', 'PMC_historical', 'fulltext_corpus/textPMCHistorical.csv')

# remove_duplicates()
# remove_non_phylogenetics_articles()

# training_test_full_texts('binary_labeled_examples.csv', 'fulltext_corpus/textcrawledData.csv',
#                          'fulltext_corpus/textiGEMsPDFs.csv')
