import glob
import pandas as pd
import textract
import ocrmypdf as ocr1
import re
import os
from utils import clean_ids
import tarfile
import zipfile
import shutil


def unzip():
    dirs_with_zipped_data = ['iGEM', 'google_scholar', 'journals']
    for dir in dirs_with_zipped_data:
        for file in os.listdir(dir):
            if file.endswith('.zip'):
                path = os.path.join(dir, file)
                with zipfile.ZipFile(path) as zip_ref:
                    zip_ref.extractall()
    dirs_with_tar_data = ['PMC', 'PMC_historical']
    for dir in dirs_with_tar_data:
        extracted = []
        files = os.listdir(dir)
        files.reverse()
        for file in files:
            if file.endswith('.tar.gz'):
                with tarfile.open(os.path.join(dir, file), "r:gz") as tar_ref:
                    for member in tar_ref.getmembers():
                        member.name = member.name.split('/')[-1]
                        extracted.append(member.name)
                        tar_ref.extract(member, os.path.join(dir, 'text_papers'))


def process_igem_data():
    csv_name = 'iGEM/iGEMsPDFs.csv'
    folder = 'iGEM/iGEMsPDFs'
    df = pd.read_csv(csv_name)
    df = df.drop_duplicates(keep=False)
    df = df.reset_index(drop=True)
    df['downloaded'] = [0] * df.shape[0]
    df['text'] = [''] * df.shape[0]
    for file in glob.glob(folder + '*.pdf'):
        id = file[file.index('/') + 1: file.index('.pdf')]
        try:
            if df[df['ID'] == id].shape[0] > 0:
                row = df[df['ID'] == id]
                try:
                    ocr1.ocr(file, folder + "output.pdf", deskew=True, force_ocr=True)
                    text = textract.process(folder + "output.pdf", method='pdftotext').decode('utf-8')
                except Exception:
                    text = textract.process(file, method='pdftotext').decode('utf-8')
                row['text'] = text
                if len(text) > 100:
                    row['downloaded'] = 1
                row = row[['ID', 'text', 'downloaded', 'Binary']]
                row.to_csv('text' + csv_name, mode='a', index=False, header=False)
            else:
                os.remove(file)
                print('File deleted: ' + file)

        except Exception as e:
            print(id, df[df['ID'] == id]['Binary'].tolist()[0], e)


def process_crawled_data(csv_name, folder, pdf):
    print(folder)
    csv_path = os.path.join(folder.split('/')[0], csv_name)
    df = pd.read_csv(csv_path)
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
                if df.at[df.index[df['ID'] == id][0], 'downloaded'] == 1:   # remove duplicate files
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
                row.to_csv('text' + csv_name, mode='a', index=False, header=False)
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


def process_biorxivTDM(csv_name, folder, new_folder, filename):
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
                        row.to_csv(filename, mode='a', index=False, header=False)
                else:  # remove files which are not related to phylogenetics
                    os.remove(file)
                    print('File deleted: ' + file)
        except Exception as e:
            print(id, e)
    shutil.move(folder, new_folder)


def process_PMC(folder, new_folder, csv_file):
    filename = os.path.join(new_folder, csv_file)
    for file in glob.glob(os.path.join(folder, '*.txt')):
        id = file.split('/')[-1].replace('.txt', '')
        try:
            with open(file, 'r', encoding='latin1') as text_file:
                text = text_file.read()
                if is_phylogenetics_text(text):
                    row = pd.DataFrame([[id, text, 1]], columns=['ID', 'text', 'downloaded'])
                    row.to_csv(filename, mode='a', index=False, header=False)
                else:  # remove files which are not related to phylogenetics
                    os.remove(file)
                    print('File deleted: ' + file)
        except Exception as e:
            print(id, e)
    shutil.move(folder, new_folder)


def remove_non_phylogenetics_articles():
    for file in glob.glob('fulltext_corpus/*.csv'):
        df = pd.read_csv(file, header=None)
        print(file, df.shape[0])
        indices_drop = []
        for index, row in df.iterrows():
            if not is_phylogenetics_text(row[1]):
                indices_drop.append(index)
        df = df.drop(labels=indices_drop, axis=0)
        print(file, len(indices_drop))
        df.to_csv(file, header=False, index=False)


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
        df1.to_csv(file, header=False, index=False)


def training_test_full_texts(labeled_file, crawler_file, igem_file):
    df = clean_ids(pd.read_csv(labeled_file))
    data = clean_ids(pd.read_csv(igem_file, header=None)[[0, 1, 2]], False)
    data2 = pd.read_csv(crawler_file, header=None)
    data = pd.concat([data, data2])
    data = data.merge(df, left_on=0, right_on='ID')[['ID', 'PY', 1, 'Binary']]
    data = data.sample(frac=1)
    df_test = data[0:int(0.2 * data.shape[0])]
    df_training = data[int(0.2 * data.shape[0]):]
    df_test.to_csv('test.csv', index=False, header=False)
    df_training.to_csv('train.csv', index=False, header=False)



os.chdir('data')
# Unzips files from different datasets
# unzip()

# To improve consistency we can clean CSV files data before processing them, but it's not necessary
# first ; needs to be removed from titles of crawledDataRound2, Gene, OABioLinnen, OABotLinnen, OABotAnnals, OAMolBio,
# OASysBio, Wiley Organic, Wiley PhytologistTrust journals, then Journals names should be connected in crawledDataRound2
# Then order of columns should be standardized in all .csv files to Title, Authors, Link, Published, Date, version,
# volume, issue, category, search criteria, iGEM label columns and headers need to be added where missing

# process_igem_data()
# process_crawled_data('crawledData_metadata.csv', 'google_scholar/downloadedPDFs/', '')
# process_crawled_data('crawledDataRound2_metadata.csv', 'google_scholar/downloadedPDFsRound2/', '')
# process_crawled_data('GeneJournalData_metadata.csv', 'journals/downloadedScienceDirectGenePDFs/', ' pdf')
# process_crawled_data('MolecularJournalData_metadata.csv', 'journals/scienceDirectMolecularPdfs/', ' pdf')
# process_crawled_data('WileyBiogeographyJournalData_metadata.csv', 'journals/downloadedWileyBiogeography/', '')
# process_crawled_data('WileyCladisticsJournalData_metadata.csv', 'journals/downloadedWileyCladistics/', '')
# process_crawled_data('WileyEcologyEvolutionJournalData_metadata.csv', 'journals/downloadedWileyEcologyEvolution/', '')
# process_crawled_data('WileyMolecularEcologyJournalData_metadata.csv', 'journals/downloadedWileyMolecularEcology/', '')
# process_crawled_data('WileyEvolutionaryBiologyJournalData_metadata.csv', 'journals/downloadedWileyEvolutionaryBiology/', '')
# process_crawled_data('WileyOrganicEvolutionJournalData_metadata.csv', 'journals/downloadedWileyOrganicEvolution/', '')
# process_crawled_data('WileyPhytologistTrustJournalData_metadata.csv', 'journals/downloadedWileyPhytologistTrust/', '')
# process_crawled_data('WileySystematicEntomologyJournalData_metadata.csv', 'journals/downloadedWileySystematicEntomology/', '')
# process_crawled_data('WileyZoologicalResearchJournalData_metadata.csv', 'journals/downloadedWileyZoologicalResearch/', '')
# process_crawled_data('OABiologicalLinneanJournalData_metadata.csv', 'journals/downloadedOABiologicalLinnean/', '')
# process_crawled_data('OABotanicalLinneanJournalData_metadata.csv', 'journals/downloadedOABotanicalLinnean/', '')
# process_crawled_data('OABotanyAnnalsJournalData_metadata.csv', 'journals/downloadedOABotanyAnnals/', '')
# process_crawled_data('OASystematicBiologyJournalData_metadata.csv', 'journals/downloadedOASystematicBiology/', '')
# process_crawled_data('OAZoologicalLinneanJournalData_metadata.csv', 'journals/downloadedOAZoologicalLinnean/', '')
# process_biorxivTDM('biorxivTDM_metadata.csv', '../../biorxivTDM', 'biorxivTDM', 'fulltext_corpus/textBiorxivTDM.csv')
# process_PMC('../../text_papers', 'PMC', 'fulltext_corpus/textPMC.csv')
# process_PMC('../../PMC_historical_text_papers', 'PMC_historical', 'fulltext_corpus/textPMCHistorical.csv')

# remove_duplicates()
# remove_non_phylogenetics_articles()

# training_test_full_texts('binary_labeled_examples.csv', 'textcrawledData.csv', 'textiGEMsPDFs.csv')

