import csv

import pandas as pd
import os

os.chdir('data/iGEM')

# updates iGEMsPDFs_metadata with new information
df1 = pd.read_csv('iGEMsPDFs_metadata.csv')
# Usable column is 3 if time data is found in both paper and supplement, 2 if just in figures in paper, 1 if it's found
# in supplement, tables, texts, but not in figures and 0 otherwise
df2 = pd.read_csv('iGEMsPositivePDFs.csv', dtype=object).replace([None], [''], regex=True)
all_positives = set(df2['i_citation_id'][df2['i_citation_id'] != '']).union(set(df2['ref_id'][df2['ref_id'] != '']))
old_positives = set(df1['ID'])
new_positives = all_positives - old_positives
not_positive_anymore = old_positives - all_positives    # old positives that are not among newly given data
print(new_positives)
print(not_positive_anymore)
labeled_positive_by_me_but_not_in_new_data = set(df1.apply(lambda x: x['ID'] if (x['ID'] in not_positive_anymore
                                                                                  and x['Usable'] > 1) else '', axis=1))
print(labeled_positive_by_me_but_not_in_new_data)

df2['New'] = df2.apply(lambda x: 1 if (x['i_citation_id'] in new_positives or x['ref_id'] in new_positives) else 0,
                       axis=1)
df3 = df2[df2['New'] == 1]
df3['ID'] = df3.apply(lambda x: x['i_citation_id'] if x['i_citation_id'] != '' else x['ref_id'], axis=1)
df3 = df3.rename(columns={'c_title': 'TI', 'c_first_author_lname': 'AU', 'i_year': 'PY'})
df3 = df3.reset_index(drop=True)
df3['JO'] = pd.Series(list(['']*df3.shape[0]))
# Usable and good figure I've labeled manually for old data, so I'll do the same for this one once metadata is updated
df3['Usable'] = pd.Series(list([-1]*df3.shape[0]))
df3['Good_figure'] = pd.Series(list([-1]*df3.shape[0]))
df3['Good_figure_2'] = pd.Series(list([-1]*df3.shape[0]))
df3['Good_figure_3'] = pd.Series(list([-1]*df3.shape[0]))
df3['Good_figure_4'] = pd.Series(list([-1]*df3.shape[0]))
df3['Good_figure_5'] = pd.Series(list([-1]*df3.shape[0]))
df3['Binary'] = pd.Series(list([1]*df3.shape[0]))
df3 = df3[['ID', 'AU', 'TI', 'JO', 'PY', 'Usable', 'Good_figure', 'Good_figure_2', 'Good_figure_3', 'Good_figure_4',
           'Good_figure_5', 'Binary']]

df1['Binary'] = df1.apply(lambda x: 1 if (x['ID'] in all_positives or x['Usable'] > 1) else 0, axis=1)


df = pd.concat([df1, df3])
print(df.shape, df1.shape, df3.shape)
df.to_csv('iGEMsPDFs_metadata.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)


# updates binary_labeled_examples with new information
df1 = pd.read_csv('iGEMsPDFs_metadata.csv')
df2 = pd.read_csv('../binary_labeled_examples.csv')
print(df1.shape, df2.shape)
df1['Dataset'] = pd.Series(list(['iGEM']*df1.shape[0]))
df1 = df1[['ID', 'PY', 'Dataset', 'Binary']]
df1 = df1.rename(columns={'Binary': 'Binary'})
df2 = df2[df2['Dataset'] == 'crawler']
df1 = pd.concat([df1, df2])
print(df1.shape, df2.shape)
df1.to_csv('../binary_labeled_examples.csv', index=False)