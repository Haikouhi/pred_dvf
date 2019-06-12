import pandas as pd
from constantes import *

class Data:

    def __init__(self, file1, file2):

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        self.df = pd.merge(df1, df2, how='outer')

    def remove_after_dot(self, string):
        return string.split('.')[0].strip()


    def convert_code_postale(self):
        self.df['code_postal'] = self.df['code_postal'].astype(str)
        self.df['code_postal'] = self.df['code_postal'].apply(self.remove_after_dot)

    def tri_code_postale(self, codes_postals):
        self.convert_code_postale()
        self.df = self.df[self.df['code_postal'].isin(codes_postals)]
        
    def tri_columns(self):
        self.df = self.df[self.df['type_local'] == 'Appartement']
        self.df = self.df.loc[(self.df['surface_reelle_bati'] <= 255) & (self.df['surface_reelle_bati'] > 10), :]
        self.df = self.df.loc[(self.df['valeur_fonciere'] >= 10000) & (self.df['valeur_fonciere'] <= 1600000), :]
        self.df = self.df[['valeur_fonciere', 'code_postal', 'type_local', 'surface_reelle_bati', 
                                             'nombre_pieces_principales', 'latitude', 'longitude']]
        self.delete_useless_rows()

    def create_column_surface_price(self):
        self.df['prix_m2'] = round(self.df.valeur_fonciere / self.df.surface_reelle_bati)
        self.df = self.df.loc[(self.df['prix_m2'] >= 1000) & (self.df['prix_m2'] <= 10000)]

    def delete_useless_rows(self):
        self.df = self.df.dropna(how='any')
        self.df = self.df.drop(columns=['type_local'])

    def delete_dirty_data(self):
        self.df = self.df[~((self.df['surface_reelle_bati'] < 128) & (self.df['valeur_fonciere'] > 1000000))]
        self.df = self.df[~((self.df['surface_reelle_bati'] > 142) & (self.df['valeur_fonciere'] < 352762))]
        self.df = self.df[~((self.df['surface_reelle_bati'] < 60) & (self.df['valeur_fonciere'] > 657000))]


